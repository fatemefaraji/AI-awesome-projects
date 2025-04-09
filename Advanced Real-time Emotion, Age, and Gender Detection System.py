import cv2
import numpy as np
import tensorflow as tf
from time import time
import os

class RealTimeFaceAnalyzer:
    def __init__(self):
        # Initialize models and parameters
        self.initialize_models()
        self.setup_labels()
        self.setup_video_capture()
        self.performance_metrics = {'fps': 0, 'last_time': time(), 'frame_count': 0}
        
    def initialize_models(self):
        """Load all required models and classifiers"""
        # Verify model files exist
        model_files = {
            'haar': 'haarcascades_models/haarcascade_frontalface_default.xml',
            'emotion': 'emotion_detection_model_100epochs_no_opt.tflite',
            'age': 'age_detection_model_50epochs_no_opt.tflite',
            'gender': 'gender_detection_model_50epochs_no_opt.tflite'
        }
        
        for key, path in model_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load Haar Cascade classifier
        self.face_classifier = cv2.CascadeClassifier(model_files['haar'])
        if self.face_classifier.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        
        # Initialize TFLite interpreters
        self.models = {
            'emotion': self.load_tflite_model(model_files['emotion'], (1, 48, 48, 1)),
            'age': self.load_tflite_model(model_files['age'], (1, 200, 200, 3)),
            'gender': self.load_tflite_model(model_files['gender'], (1, 200, 200, 3))
        }
    
    def load_tflite_model(self, model_path, expected_input_shape):
        """Helper function to load a TFLite model with shape verification"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        actual_shape = tuple(input_details[0]['shape'])
        
        if actual_shape != expected_input_shape:
            raise ValueError(f"Model input shape mismatch. Expected {expected_input_shape}, got {actual_shape}")
        
        return {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': interpreter.get_output_details()
        }
    
    def setup_labels(self):
        """Initialize all classification labels"""
        self.labels = {
            'emotion': ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            'gender': ['Male', 'Female']
        }
    
    def setup_video_capture(self):
        """Initialize video capture with optimized settings"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Set optimal camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def preprocess_face(self, face_img, target_size, grayscale=False):
        """Preprocess face image for model input"""
        # Resize to target dimensions
        face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        
        if grayscale:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
        else:
            face_img = face_img.astype('float32')
        
        return np.expand_dims(face_img, axis=0)  # Add batch dimension
    
    def predict(self, model_name, input_data):
        """Run inference using the specified model"""
        model = self.models[model_name]
        
        # Verify input shape matches model expectation
        expected_shape = tuple(model['input_details'][0]['shape'])
        if input_data.shape != expected_shape:
            raise ValueError(f"Input shape {input_data.shape} doesn't match model expectation {expected_shape}")
        
        model['interpreter'].set_tensor(model['input_details'][0]['index'], input_data)
        model['interpreter'].invoke()
        return model['interpreter'].get_tensor(model['output_details'][0]['index'])
    
    def draw_results(self, frame, faces, results):
        """Draw bounding boxes and labels on the frame"""
        for (x, y, w, h), result in zip(faces, results):
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Create combined label text
            label_text = f"{result['emotion']} | {result['gender']} | {result['age']}y"
            
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw performance info
        cv2.putText(frame, f"FPS: {self.performance_metrics['fps']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    def update_performance_metrics(self):
        """Calculate and update FPS using smoothed average"""
        self.performance_metrics['frame_count'] += 1
        
        # Update FPS every 10 frames for smoother display
        if self.performance_metrics['frame_count'] % 10 == 0:
            current_time = time()
            elapsed = current_time - self.performance_metrics['last_time']
            self.performance_metrics['fps'] = 10 / elapsed
            self.performance_metrics['last_time'] = current_time
    
    def process_frame(self, frame):
        """Process a single frame for face analysis"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_classifier.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        for (x, y, w, h) in faces:
            try:
                # Get face regions with boundary checks
                y1, y2 = max(0, y), min(y + h, frame.shape[0])
                x1, x2 = max(0, x), min(x + w, frame.shape[1])
                
                face_gray = gray[y1:y2, x1:x2]
                face_color = frame[y1:y2, x1:x2]
                
                # Skip if face region is too small
                if face_gray.size == 0 or face_color.size == 0:
                    continue
                
                # Preprocess for each model
                emotion_input = self.preprocess_face(face_gray, (48, 48), grayscale=True)
                demo_input = self.preprocess_face(face_color, (200, 200))
                
                # Run predictions
                emotion_pred = self.predict('emotion', emotion_input)
                gender_pred = self.predict('gender', demo_input)
                age_pred = self.predict('age', demo_input)
                
                # Process results
                results.append({
                    'emotion': self.labels['emotion'][np.argmax(emotion_pred)],
                    'gender': self.labels['gender'][int(gender_pred[0][0] >= 0.5)],
                    'age': str(int(age_pred[0][0]))
                })
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return faces, results
    
    def run(self):
        """Main loop for real-time face analysis"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                faces, results = self.process_frame(frame)
                
                # Display results if any faces detected
                if len(faces) > 0:
                    self.draw_results(frame, faces, results)
                
                self.update_performance_metrics()
                
                # Show frame
                cv2.imshow('Face Analysis Dashboard', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        analyzer = RealTimeFaceAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")