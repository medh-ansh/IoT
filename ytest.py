import os, time, csv, cv2, board, adafruit_scd4x, busio, spidev
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import pytesseract
import re

i2c = busio.I2C(board.D3, board.D2)
sensor = adafruit_scd4x.SCD4X(i2c)
sensor.start_periodic_measurement()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# Load both models: yolo26n for general objects, best.onnx for dedicated number plate detection
session = ort.InferenceSession("yolo26n.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

plate_session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
plate_input_name = plate_session.get_inputs()[0].name

BIKE_CLASS_IDS = {1, 3}
VEHICLE_CLASS_IDS = {5, 7}
CSV_FILE_PATH = "/home/raspberry/Desktop/data.csv"

try:
    while True:
        try:
            temp, hum, co2 = sensor.temperature or 0.0, sensor.relative_humidity or 0.0, sensor.CO2 or 0.0
            
            # Capture frame and prepare it for the model structure
            img = picam2.capture_array()
            input_data = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
            input_data = np.transpose(input_data, (2, 0, 1)) # Change HWC to CHW format
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

            # Run inference - cleanly parsed outputs are returned natively
            outputs = session.run(None, {input_name: input_data})
            output_array = np.squeeze(outputs[0]) # Yields clean shape: (84, 8400)
            
            # Parse the bounding anchors cleanly using standard YOLO matrix math
            human_present = 0
            vehicle_present = 0
            car_present = 0
            bike_present = 0
            numberplate = "N/A"
            
            # 1. Check for Humans (Class ID: 0)
            human_scores = output_array[4 + 0, :]
            if np.any(human_scores > 0.75):
                human_present = 1
            
            car_scores = output_array[4 + 2, :]
            if np.any(car_scores > 0.6):
                car_present = 1
                 
            for class_id in BIKE_CLASS_IDS:
               bike_scores = output_array[4 + class_id, :]
               if np.any(bike_scores > 0.6):
                   bike_present = 1
                   break

            # 2. Check for Vehicles (Class IDs: 5, 7 for bus and truck and stuff)
            for class_id in VEHICLE_CLASS_IDS:
                class_scores = output_array[4 + class_id, :]
                if np.any(class_scores > 0.75):
                    vehicle_present = 1
                    break                     

            # Dedicated two-stage number plate detection using best.onnx
            h_img, w_img, _ = img.shape
            plate_input = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
            plate_input = np.transpose(plate_input, (2, 0, 1))
            plate_input = np.expand_dims(plate_input, axis=0)

            plate_outputs = plate_session.run(None, {plate_input_name: plate_input})
            plate_output_array = np.squeeze(plate_outputs[0])
            plate_preds = np.transpose(plate_output_array, (1, 0))

            best_plate_conf = 0.0
            best_plate_box = None

            for box in plate_preds:
                box_coords = box[0:4]
                class_scores = box[4:]
                cls_id = np.argmax(class_scores)
                score = class_scores[cls_id]

                if score > 0.75 and score > best_plate_conf:
                    best_plate_conf = score
                    best_plate_box = box_coords

            if best_plate_box is not None:
                xc, yc, w, h = best_plate_box
                scale_x = w_img / 640.0
                scale_y = h_img / 640.0

                x1 = max(0, int((xc * scale_x) - (w * scale_x / 2)))
                y1 = max(0, int((yc * scale_y) - (h * scale_y / 2)))
                x2 = min(w_img, int((xc * scale_x) + (w * scale_x / 2)))
                y2 = min(h_img, int((yc * scale_y) + (h * scale_y / 2)))

                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    
                    # 1. Denoise while preserving sharp letter boundaries
                    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
                    
                    # 2. Upscale sharply for the OCR engine
                    resized = cv2.resize(filtered, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                    
                    # 3. Adaptive thresholding handles uneven lighting/shadows much better than global Otsu
                    thresh_plate = cv2.adaptiveThreshold(
                        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # 4. Morphological closing to seal broken character lines
                    kernel = np.ones((2, 2), np.uint8)
                    processed_plate = cv2.morphologyEx(thresh_plate, cv2.MORPH_CLOSE, kernel)
                    
                    # Try PSM 8 (Single Word) for strict character grouping, fallback to PSM 7
                    for psm_mode in [8, 7]:
                        ocr_config = f'--psm {psm_mode} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        raw_text = pytesseract.image_to_string(processed_plate, config=ocr_config)
                        cleaned = "".join(raw_text.split())
                        
                        if re.match(r'^[A-Z0-9]{4,10}$', cleaned):
                            numberplate = cleaned
                            break

            timestamp = time.ctime()
            print(f"[{timestamp}] Temp: {temp:.2f}°C | Hum: {hum:.2f}% | CO2: {co2:.2f}ppm | Car: {car_present} | Bike: {bike_present} | Heavy vehicle: {vehicle_present} | Human: {human_present} | Plate: {numberplate}")
            
            RST = 25

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(RST, GPIO.OUT, initial=GPIO.HIGH)

            spi = spidev.SpiDev()
            spi.open(0, 0)
            spi.max_speed_hz = 500000
            spi.mode = 0

            def write_reg(address, value):
                spi.xfer2([address | 0x80, value & 0xFF])

            def read_reg(address):
                return spi.xfer2([address & 0x7F, 0x00])[1]

            try:
                # Reset radio
                GPIO.output(RST, GPIO.LOW)
                time.sleep(0.02)
                GPIO.output(RST, GPIO.HIGH)
                time.sleep(0.05)

                version = read_reg(0x42)
                print("SX1278 version:", hex(version))

                if version != 0x12:
                    raise RuntimeError("SX1278 not detected correctly")

                # LoRa sleep, then standby
                write_reg(0x01, 0x80)
                time.sleep(0.01)
                write_reg(0x01, 0x81)
                time.sleep(0.01)

                # Frequency: 433 MHz
                write_reg(0x06, 0x6C)
                write_reg(0x07, 0x40)
                write_reg(0x08, 0x00)

                # SF7, BW 125 kHz, CR 4/5, explicit header, CRC enabled
                write_reg(0x1D, 0x72)
                write_reg(0x1E, 0x74)
                write_reg(0x26, 0x04)

                # Preamble length: 8 symbols
                write_reg(0x20, 0x00)
                write_reg(0x21, 0x08)

                # TX FIFO starts at address 0
                write_reg(0x0E, 0x00)
                write_reg(0x0D, 0x00)

                # Conservative PA_BOOST setting for this initial test
                write_reg(0x09, 0x88)

                # Clear pending IRQ flags
                write_reg(0x12, 0xFF)
                
                message = f"DATA, [{timestamp}] Temp: {temp:.2f}°C | Hum: {hum:.2f}% | CO2: {co2:.2f}ppm | Car: {car_present} | Bike: {bike_present} | Heavy vehicle: {vehicle_present} | Human: {human_present} | Plate: {numberplate}".encode('utf-8')
                write_reg(0x0D, 0x00)

                for byte in message:
                    write_reg(0x00, byte)

                write_reg(0x22, len(message))

                print("Sending test packet...")
                write_reg(0x12, 0xFF)
                write_reg(0x01, 0x83)  # LoRa TX mode

                deadline = time.time() + 5

                while time.time() < deadline:
                    if read_reg(0x12) & 0x08:  # TxDone
                        write_reg(0x12, 0x08)
                        write_reg(0x01, 0x81)  # Return to standby
                        break
                    time.sleep(0.01)
                else:
                    write_reg(0x01, 0x81)
                    print("TX timeout: TxDone flag not observed")

            finally:
                spi.close()
                GPIO.cleanup()
                        
            with open(CSV_FILE_PATH, mode='w', newline='') as datafile:
                writer = csv.writer(datafile)
                writer.writerow(["Timestamp", "Temperature", "CO2", "Humidity", "Car_Present", "Bike_Present", "Heavy_Vehicle_Present", "Human_Present", "Number_Plate"])
                writer.writerow([timestamp, temp, co2, hum, car_present, bike_present, vehicle_present, human_present, numberplate])

        except KeyboardInterrupt as error:
            print(f"Sensor drop: {error.args}")
        time.sleep(10)
finally:
    sensor.exit()
    picam2.stop()
    picam2.close()
