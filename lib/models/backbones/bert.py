import os

from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Bert(nn.Module):
    def __init__(self, pool):
        super().__init__()

        self.pool = pool
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.config = BertConfig.from_pretrained(
            "pretrained/bert-base-uncased/bert_config.json"
        )
        self.model = BertModel.from_pretrained(
            "pretrained/bert-base-uncased/pytorch_model.bin",
            config=self.config,
            add_pooling_layer=self.pool,
        )
        self.out_channels = 768

    def forward(self, captions):
        texts = [caption.text for caption in captions]
        tokens = self.tokenizer(
            texts, max_length=100, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].cuda()
        attention_mask = tokens["attention_mask"].cuda()
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        if self.pool:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]


def build_bert(cfg):
    model = Bert(cfg.MODEL.BERT.POOL)
    model.train()

    if cfg.MODEL.FREEZE:
        pass

    return model
