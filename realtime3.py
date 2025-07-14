import pickle
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        # Initialize text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Detection parameters
        self.min_meaningful_length = 20  # Minimum chars after preprocessing
        self.confidence_threshold = 0.7  # Minimum confidence to accept prediction
        self.min_meaningful_words = 5    # Minimum words after preprocessing
        
        # Fake news keywords and patterns
        self.fake_keywords = {
            'urgent','shocking', 'whistleblower', 'leaked',
            'cover-up', 'they dont want you to know', 'hidden truth',
            '5g', 'microchip', 'bill gates', 'population control',
            'alien', 'dormant nanobots', 'implant', 'exposed', 'big pharma'
        }
        
        # Load models
        try:
            with open("C:\\Users\\Sudheesh\\OneDrive\\Desktop\\LogisticRegression\\fake_news_model.pkl", "rb") as model_file:
                self.model = pickle.load(model_file)
            with open("C:\\Users\\Sudheesh\\OneDrive\\Desktop\\LogisticRegression\\count_vectorizer.pkl", "rb") as vectorizer_file:
                self.count_vectorizer = pickle.load(vectorizer_file)
            with open("tfidf_transformer.pkl", "rb") as transformer_file:
                self.tfidf_transformer = pickle.load(transformer_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model files: {str(e)}")
            raise

    def contains_fake_patterns(self, text):
        """Check for obvious fake news patterns"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.fake_keywords)

    def is_gibberish(self, text):
        """Check if text appears to be random/nonsensical"""
        if not text:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        unique_ratio = len(set(text)) / len(text)
        return alpha_ratio < 0.5 or unique_ratio > 0.7

    def preprocess_text(self, text):
        """Enhanced text preprocessing with validation"""
        if not isinstance(text, str) or not text.strip():
            return None
            
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        processed_text = ' '.join(words)
        
        # Validate the processed text
        if (len(processed_text) < self.min_meaningful_length or 
            len(words) < self.min_meaningful_words or
            self.is_gibberish(processed_text)):
            return None
            
        return processed_text

    def predict(self, text_list):
        """Safe prediction with enhanced validation"""
        if not text_list:
            return []
            
        results = []
        for original_text in text_list:
            # First check for obvious fake patterns
            if self.contains_fake_patterns(original_text):
                results.append({
                    'text': original_text,
                    'prediction': 'Fake',
                    'confidence': 0.99,
                    'reason': 'Contains known fake news patterns',
                    'is_reliable': True,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
                
            processed_text = self.preprocess_text(original_text)
            
            if not processed_text:
                results.append({
                    'text': original_text,
                    'error': 'Text too short, invalid, or appears random',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
                
            try:
                # Transform and predict
                count_vector = self.count_vectorizer.transform([processed_text])
                tfidf_vector = self.tfidf_transformer.transform(count_vector)
                prediction = self.model.predict(tfidf_vector)[0]
                probabilities = self.model.predict_proba(tfidf_vector)[0]
                confidence = max(probabilities)
                
                # Only accept predictions above confidence threshold
                if confidence < self.confidence_threshold:
                    results.append({
                        'text': original_text,
                        'error': f'Low confidence prediction ({confidence*100:.1f}%)',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue
                    
                # 0 = Fake, 1 = Real
                results.append({
                    'text': original_text,
                    'processed_text': processed_text,
                    'prediction': 'Fake' if prediction == 0 else 'Real',
                    'confidence': float(confidence),
                    'fake_prob': float(probabilities[0]),
                    'real_prob': float(probabilities[1]),
                    'is_reliable': confidence >= self.confidence_threshold,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                results.append({
                    'text': original_text,
                    'error': f'Prediction failed: {str(e)}',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
        return results

class FakeNewsDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fake News Detector")
        
        # Start in full screen
        self.attributes('-fullscreen', True)
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)
        
        # Set initial font sizes
        self.base_font_size = 14
        self.header_font_size = 18
        
        try:
            self.detector = FakeNewsDetector()
        except Exception as e:
            self.destroy()
            return
            
        self.create_widgets()
        
        # Configure window resizing
        self.bind('<Configure>', self.on_window_resize)
    
    def toggle_fullscreen(self, event=None):
        self.attributes('-fullscreen', not self.attributes('-fullscreen'))
    
    def exit_fullscreen(self, event=None):
        self.attributes('-fullscreen', False)
    
    def on_window_resize(self, event):
        # Adjust font sizes based on window height
        window_height = self.winfo_height()
        self.base_font_size = max(12, window_height // 45)
        self.header_font_size = max(16, window_height // 35)
        
        # Update all widget fonts
        self.update_font_sizes()
    
    def update_font_sizes(self):
        # Update style configurations
        self.style.configure("TButton", 
                           font=('Arial', self.base_font_size), 
                           padding=6)
        self.style.configure("TLabel", 
                           font=('Arial', self.base_font_size))
        self.style.configure("Header.TLabel", 
                           font=('Arial', self.header_font_size, 'bold'))
        
        # Update text widgets
        self.input_text.configure(font=('Arial', self.base_font_size))
        self.results_text.configure(font=('Consolas', self.base_font_size))
    
    def create_widgets(self):
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TButton", 
                           font=('Arial', self.base_font_size), 
                           padding=6)
        self.style.configure("TLabel", 
                           font=('Arial', self.base_font_size))
        self.style.configure("Header.TLabel", 
                           font=('Arial', self.header_font_size, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(main_frame, text="Fake News Detector", style="Header.TLabel")
        header.pack(pady=20)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Enter News Article to Analyze", padding="15")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            width=80, 
            height=12, 
            wrap=tk.WORD,
            font=('Arial', self.base_font_size),
            padx=10,
            pady=10
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        analyze_btn = ttk.Button(
            btn_frame, 
            text="Analyze", 
            command=self.analyze_text,
            style="TButton"
        )
        analyze_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = ttk.Button(
            btn_frame, 
            text="Clear", 
            command=self.clear_all,
            style="TButton"
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        exit_fullscreen_btn = ttk.Button(
            btn_frame,
            text="Exit Fullscreen",
            command=self.exit_fullscreen,
            style="TButton"
        )
        exit_fullscreen_btn.pack(side=tk.RIGHT, padx=10)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            width=80, 
            height=15, 
            wrap=tk.WORD,
            font=('Consolas', self.base_font_size),
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for colored text
        self.results_text.tag_config("fake", foreground="red")
        self.results_text.tag_config("real", foreground="green")
        self.results_text.tag_config("error", foreground="orange")
        self.results_text.tag_config("meta", foreground="blue")
    
    def analyze_text(self):
        """Analyze the entered text"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze")
            return
        
        results = self.detector.predict([text])
        self.display_results(results)
    
    def display_results(self, results):
        """Display analysis results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No results to display\n", "error")
        else:
            for result in results:
                # Add timestamp
                self.results_text.insert(tk.END, f"[{result.get('timestamp')}]\n", "meta")
                
                if 'error' in result:
                    self.results_text.insert(tk.END, f"Error: {result['error']}\n", "error")
                elif 'reason' in result:
                    self.results_text.insert(tk.END, f"Detection: {result['reason']}\n", "meta")
                    self.results_text.insert(tk.END, f"Prediction: {result['prediction']}\n", result['prediction'].lower())
                else:
                    # Prediction line
                    pred_text = f"Prediction: {result['prediction']} "
                    self.results_text.insert(tk.END, pred_text, result['prediction'].lower())
                    
                    # Confidence
                    conf_text = f"(Confidence: {result['confidence']*100:.1f}%)\n"
                    self.results_text.insert(tk.END, conf_text)
                    
                    # Probabilities
                    prob_text = f"Fake: {result['fake_prob']*100:.1f}% | Real: {result['real_prob']*100:.1f}%\n"
                    self.results_text.insert(tk.END, prob_text)
                
                # Original text excerpt
                if 'text' in result:
                    self.results_text.insert(tk.END, f"Text: {result['text'][:200]}...\n")
                
                self.results_text.insert(tk.END, "-"*80 + "\n\n")
        
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)
    
    def clear_all(self):
        """Clear both input and results"""
        self.input_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = FakeNewsDetectorApp()
    app.mainloop()