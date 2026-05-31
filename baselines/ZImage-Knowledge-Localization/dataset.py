"""
Dataset classes for ZImage knowledge localization
Adapted from DiT-Knowledge-Localization for ZImage
"""
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset


# Use the dataset folder in parent directory
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent


def get_knowledge_dataset_class_and_get_list_fn(knowledge_type, for_model=None):
    """Get dataset class and knowledge list function for given knowledge type"""
    if knowledge_type == "style":
        if for_model is None:
            return StyleDataset, get_artists_list
        else:
            return StyleDataset, partial(get_artists_list_for_model, model_name=for_model)
    elif knowledge_type == "place":
        return PlacesDataset, get_places_list
    elif knowledge_type == "safety":
        return SafetyDataset, get_safety_list
    elif knowledge_type == "celebrity":
        return CelebrityDataset, get_celebrities_list
    elif knowledge_type == "animal":
        return AnimalDataset, get_animals_list
    elif knowledge_type == "copyright":
        return CopyrightDataset, get_copyrights_list
    else:
        raise ValueError(f"Knowledge type {knowledge_type} not supported")


def get_eval_text_for_knowledge(knowledge_type, knowledge):
    """Get evaluation text for CLIP score calculation"""
    if knowledge_type == "style":
        return f"a photo in the style of {knowledge}"
    elif knowledge_type == "place":
        return f"a photo of {knowledge}"
    elif knowledge_type == "safety":
        return f"a photo of {knowledge}"
    elif knowledge_type == "celebrity":
        return f"a photo of {knowledge}"
    elif knowledge_type == "animal":
        return f"a photo of {knowledge}"
    elif knowledge_type == "copyright":
        return f"a photo of {knowledge}"
    else:
        raise ValueError(f"Knowledge type {knowledge_type} not supported")


def _get_stripped_lines_list_from_file(file_path):
    """Read lines from file and strip whitespace"""
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


class BaseDataset(Dataset, ABC):
    """Base dataset class for knowledge localization"""
    def __init__(self, split, knowledge):
        super().__init__()
        
        assert split in ["train", "eval", "both"]
        
        self.splits = [split]
        if split == "both":
            self.splits = ["train", "eval"]
        
        self.knowledge = knowledge
    
    @abstractmethod
    def get_clean_prompt(self, prompt):
        """Remove knowledge reference from prompt (for intervention)"""
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass


def get_artists_list():
    """Get full list of artists"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/style/wikiart_artists.txt")


def get_artists_list_for_model(model_name):
    """Get model-specific artist list (filtered by threshold)"""
    assert model_name in ["pixart", "sana", "flux", "zimage"]
    
    if model_name == "pixart":
        file_path = f"{PROJECT_ROOT_DIR}/dataset/style/wikiart_artists_pixart_0.82_th.txt"
    elif model_name == "sana":
        file_path = f"{PROJECT_ROOT_DIR}/dataset/style/wikiart_artists_sana_0.8_th.txt"
    elif model_name == "flux":
        file_path = f"{PROJECT_ROOT_DIR}/dataset/style/wikiart_artists_flux_0.81_th.txt"
    elif model_name == "zimage":
        # Use FLUX list as default for ZImage (can create zimage-specific list later)
        file_path = f"{PROJECT_ROOT_DIR}/dataset/style/wikiart_artists_flux_0.81_th.txt"
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return _get_stripped_lines_list_from_file(file_path)


class StyleDataset(BaseDataset):
    """Dataset for style (artist) knowledge localization"""
    def __init__(self, artist: str, split: str):
        super().__init__(split, artist)
        
        self.artist = artist
        
        self.prompts = []
        for split in self.splits:
            with open(f"{PROJECT_ROOT_DIR}/dataset/style/{split}_prompts.txt", "r") as f:
                self.prompts.extend([prompt.strip() for prompt in f.readlines()])
    
    def get_clean_prompt(self, prompt):
        """Remove artist's style reference from prompt"""
        return prompt.replace(f" in the style of {self.artist}", "")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx] + " in the style of " + self.artist


def get_places_list():
    """Get list of places"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/places/places.txt")


class PlacesDataset(BaseDataset):
    """Dataset for place knowledge localization"""
    def __init__(self, place: str, split: str):
        super().__init__(split, place)
        
        self.place = place
        
        self.prompts = []
        for split in self.splits:
            self.prompts.extend(_get_stripped_lines_list_from_file(
                f"{PROJECT_ROOT_DIR}/dataset/places/{split}_prompts.txt"
            ))
    
    def get_clean_prompt(self, prompt):
        """Remove place reference from prompt"""
        return prompt.replace(f"{self.place}", "a place")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.place + " " + self.prompts[idx]


def get_safety_list():
    """Get list of safety concepts"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/safety/safety.txt")


class SafetyDataset(BaseDataset):
    """Dataset for safety concept localization"""
    def __init__(self, base_safety_prompt: str, split: str):
        super().__init__(split, base_safety_prompt)
        
        self.base_safety_prompt = base_safety_prompt
        
        self.prompts = []
        for split in self.splits:
            self.prompts.extend(_get_stripped_lines_list_from_file(
                f"{PROJECT_ROOT_DIR}/dataset/safety/{split}_prompts.txt"
            ))
    
    def get_clean_prompt(self, prompt):
        """Remove safety prompt from prompt"""
        return prompt.replace(f"{self.base_safety_prompt}", "a person")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.base_safety_prompt + self.prompts[idx]


def get_celebrities_list():
    """Get list of celebrities"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/celebrities/celebrities.txt")


class CelebrityDataset(BaseDataset):
    """Dataset for celebrity knowledge localization"""
    def __init__(self, celebrity: str, split: str):
        super().__init__(split, celebrity)
        
        self.celebrity = celebrity
        
        self.prompts = []
        for split in self.splits:
            self.prompts.extend(_get_stripped_lines_list_from_file(
                f"{PROJECT_ROOT_DIR}/dataset/celebrities/{split}_prompts.txt"
            ))
    
    def get_clean_prompt(self, prompt):
        """Remove celebrity from prompt"""
        return prompt.replace(f"{self.celebrity}", "a person")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.celebrity + " " + self.prompts[idx]


def get_animals_list():
    """Get list of animals"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/animals/animals.txt")


class AnimalDataset(BaseDataset):
    """Dataset for animal knowledge localization"""
    def __init__(self, animal: str, split: str):
        super().__init__(split, animal)
        
        self.animal = animal
        
        self.prompts = []
        for split in self.splits:
            self.prompts.extend(_get_stripped_lines_list_from_file(
                f"{PROJECT_ROOT_DIR}/dataset/animals/{split}_prompts.txt"
            ))
    
    def get_clean_prompt(self, prompt):
        """Remove animal from prompt"""
        return prompt.replace(f"{self.animal}", "an animal")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.animal + " " + self.prompts[idx]


def get_copyrights_list():
    """Get list of copyrighted characters"""
    return _get_stripped_lines_list_from_file(f"{PROJECT_ROOT_DIR}/dataset/copyrights/copyrights.txt")


class CopyrightDataset(BaseDataset):
    """Dataset for copyright character localization"""
    def __init__(self, copyright: str, split: str):
        super().__init__(split, copyright)
        
        self.copyright = copyright
        
        self.prompts = []
        for split in self.splits:
            self.prompts.extend(_get_stripped_lines_list_from_file(
                f"{PROJECT_ROOT_DIR}/dataset/copyrights/{split}_prompts.txt"
            ))
    
    def get_clean_prompt(self, prompt):
        """Remove copyright character from prompt"""
        return prompt.replace(f"{self.copyright}", "a character")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.copyright + " " + self.prompts[idx]
