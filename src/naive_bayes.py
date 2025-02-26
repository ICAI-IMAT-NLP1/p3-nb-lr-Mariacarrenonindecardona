import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        num_examples: int = labels.shape[0]
        class_counts: torch.Tensor = torch.bincount(labels.to(torch.int64))
        self.class_priors = {i: class_counts[i].item() / num_examples for i in range(len(class_counts))}
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        class_priors: Dict[int, torch.Tensor] = {}

        for label in torch.unique(labels):
            unique_label: int = int(label)
            mascara: torch.Tensor = (labels==unique_label)
            class_priors[unique_label] = len(labels[mascara])/len(labels)

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        class_word_counts: Dict[int, torch.Tensor] = {}

        for i in range(len(features)):
            bow: torch.Tensor = features[i,:]
            label: int = labels[i].item()

            if label in class_word_counts:
                class_word_counts[label] += bow
            else:
                class_word_counts[label] = bow

        conditional_probabilities: Dict[int, torch.Tensor] = {}
        for label, word_counts in class_word_counts.items():
            total_words: int = word_counts.sum()
            smoothed_probs: torch.Tensor = (word_counts + delta) / (total_words + delta * self.vocab_size)
            conditional_probabilities[label] = smoothed_probs

        return conditional_probabilities

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = torch.zeros(len(self.class_priors))
        #log_posteriors:Dict[int, float] = {}
        for class_label in self.class_priors.keys():
            log_prior: torch.Tensor = torch.log(torch.tensor(self.class_priors[class_label], dtype=torch.float32))
            conditional_prob: torch.Tensor = self.conditional_probabilities[class_label]
            log_likelihood: torch.Tensor = torch.sum(feature*torch.log(conditional_prob))
            log_posteriors[class_label] = log_prior + log_likelihood
        print(log_posteriors)
            
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """

        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
     
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        pred: int = -1

        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)

        pred: int = torch.argmax(log_posteriors).item()

        return pred
    

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """

        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
    
        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = torch.zeros(len(self.class_priors))

        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)

        probs = torch.softmax(log_posteriors, dim=0)
        return probs
