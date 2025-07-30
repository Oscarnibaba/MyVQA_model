import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .vision.vit import VisionTransformer
from .xbert import BertConfig, BertModel

class AnswerClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = self.fc2(x)
        return x

class VQA_Classifier(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 num_answer_classes=3129):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_answer_classes = num_answer_classes

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)

        # 使用改进版分类器
        self.classifier = AnswerClassifier(config_encoder.hidden_size, num_answer_classes)

    def forward(self, image, question, labels=None, train=True):
        # 图像编码
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 文本编码
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(image.device)
        question_output = self.text_encoder(
            input_ids=question.input_ids,
            attention_mask=question.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True
        )

        # 获取 [CLS] 表示
        cls_rep = question_output.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        logits = self.classifier(cls_rep)

        if train:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            return preds, probs
