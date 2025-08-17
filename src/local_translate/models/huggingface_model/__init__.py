
# 
# from abc import ABC, abstractmethod
# from typing import Dict, Any


# class HFTranslater(ABC):
#     """
#     Abstract base class for Hugging Face translation models.
#     """

#     @abstractmethod
#     def prompt_template_fn(
#         self, query: str, src_language: str, trg_language: str
#     ) -> str:
#         """
#         Generate a prompt template for translation.
        
#         :param query: The text to translate.
#         :param src_language: The source language code.
#         :param trg_language: The target language code.
#         :return: A formatted prompt string.
#         """
#         pass
    
#     @abstractmethod
#     def translate(
#         self, query: str, src_language: str, trg_language: str
#     ) -> str:
#         """
#         Translate the given text from source to target language.
        
#         :param query: The text to translate.
#         :param src_language: The source language code.
#         :param trg_language: The target language code.
#         :return: The translated text.
#         """
#         pass
    
#     @abstractmethod
#     def load_model(self, model_id: str, **config: Dict[str, Any]) -> None:
#         """
#         Load the translation model with the specified configuration.
        
#         :param model_id: The identifier for the model to load.
#         :param config: Configuration parameters for loading the model.
#         """
#         pass
    
#     @abstractmethod
#     def load_tokenizer(self, model_id: str, **config: Dict[str, Any]) -> None:
#         """
#         Load the tokenizer for the translation model.
        
#         :param model_id: The identifier for the model whose tokenizer to load.
#         """
#         pass
    
# __all__ = ["HFTranslater"]