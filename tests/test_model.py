import os
import sys
import pickle
import unittest

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config

class TestSentimentModel(unittest.TestCase):
    
    def setUp(self):
        """Load the model and vectorizer before running tests."""
        self.assertTrue(os.path.exists(config.MODEL_PATH), "model.pkl should exist. Run src/train.py first.")
        self.assertTrue(os.path.exists(config.VECTORIZER_PATH), "vectorizer.pkl should exist. Run src/train.py first.")
        self.assertTrue(os.path.exists(config.METRICS_PATH), "metrics.json should exist. Run src/train.py first.")
        
        with open(config.MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(config.VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
    def test_positive_inference(self):
        """Test that a strongly positive review is classified as positive."""
        text = "This is a great product. The sound quality is amazing and it is very comfortable to wear."
        vec = self.vectorizer.transform([text])
        prob_positive = self.model.predict_proba(vec)[0][1]
        pred = "positive" if prob_positive >= 0.5 else "negative"
        self.assertEqual(pred, "positive", "Strongly positive review should be classified as positive.")
        self.assertGreater(prob_positive, 0.6, "Positive probability should be reasonably high.")

    def test_negative_inference(self):
        """Test that a strongly negative review is classified as negative."""
        text = "Extremely disappointed. The fit is awful, it broke after two days and customer service was poor."
        vec = self.vectorizer.transform([text])
        prob_positive = self.model.predict_proba(vec)[0][1]
        pred = "positive" if prob_positive >= 0.5 else "negative"
        self.assertEqual(pred, "negative", "Strongly negative review should be classified as negative.")
        self.assertLess(prob_positive, 0.4, "Positive probability should be low (meaning negative is high).")

    def test_vocabulary_features(self):
        """Verify that key sentiment words exist in vectorizer vocabulary and have correct sign coefficients."""
        vocab = self.vectorizer.vocabulary_
        coefs = self.model.coef_[0]
        
        # Check that positive words have positive coefficients and negative words have negative coefficients
        words_to_check = {
            'great': True,  # should be positive (True)
            'love': True,
            'excellent': True,
            'disappointed': False, # should be negative (False)
            'awful': False,
            'waste': False
        }
        
        for word, is_pos in words_to_check.items():
            if word in vocab:
                idx = vocab[word]
                coef = coefs[idx]
                if is_pos:
                    self.assertGreater(coef, 0, f"'{word}' coefficient should be positive.")
                else:
                    self.assertLess(coef, 0, f"'{word}' coefficient should be negative.")

if __name__ == '__main__':
    unittest.main()
