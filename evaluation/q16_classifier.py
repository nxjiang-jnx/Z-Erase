import torch
import PIL
import pickle
import clip
import clip.clip as clip_module
import os
import warnings

# 2025.12.25:
# You should install clip from the official repository:
# pip install git+https://github.com/openai/CLIP.git

class ClipWrapper(torch.nn.Module):
    def __init__(self, device, model_name='ViT-L/14', model_path=None):
        super(ClipWrapper, self).__init__()
        
        download_root = '.cache'
        os.makedirs(download_root, exist_ok=True)
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.clip_model, self.preprocess = clip.load(model_name, device, jit=False)
            self.clip_model.load_state_dict(state_dict)
            self.clip_model.eval()
            return
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            original_download = clip_module._download
            
            def _download_patch(url: str, root: str):
                os.makedirs(root, exist_ok=True)
                filename = os.path.basename(url)
                download_target = os.path.join(root, filename)
                
                if os.path.isfile(download_target):
                    return download_target
                
                return original_download(url, root)
            
            clip_module._download = _download_patch
            try:
                self.clip_model, self.preprocess = clip.load(model_name,
                                                             device,
                                                             jit=False, download_root=download_root)
            finally:
                clip_module._download = original_download
        
        self.clip_model.eval()

    def forward(self, x):
        return self.clip_model.encode_image(x)


class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, device):
        super(SimClassifier, self).__init__()
        self.embeddings = torch.nn.parameter.Parameter(embeddings)

    def forward(self, x):
        embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1,
                                                                 keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = x / x.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_norm @ embeddings_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def initialize_prompts(clip_model, text_prompts, device):
    text = clip.tokenize(text_prompts).to(device)
    return clip_model.encode_text(text)


def save_prompts(classifier, save_path):
    prompts = classifier.embeddings.detach().cpu().numpy()
    pickle.dump(prompts, open(save_path, 'wb'))


def load_prompts(file_path, device):
    return torch.HalfTensor(pickle.load(open(file_path, 'rb'))).to(device)

def compute_embeddings(clip_model, image, device):
    images = [clip_model.preprocess(image)]
    images = torch.stack(images).to(device)
    return clip_model(images).half() 