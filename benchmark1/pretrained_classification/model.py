from transformers import (
    BertPreTrainedModel,BertModel,BertConfig,
    RobertaConfig,RobertaModel,
    AlbertPreTrainedModel,AlbertModel,AlbertConfig,
    ElectraPreTrainedModel,ElectraModel,ElectraConfig,
)
from transformers.modeling_roberta import RobertaClassificationHead
from transformers.modeling_electra import ElectraClassificationHead
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,MultiLabelSoftMarginLoss
import torch

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}
class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,0.1]).cuda())
    def __call__(self, output, target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input=output, target=target)
        return loss


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config:BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
        else:
            outputs = (0.0,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class RoBERTaForMultiLabelClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config:RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            # loss_fct = BCEWithLogLoss()
            # loss = loss_fct(logits, labels)
            # loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        else:
            outputs = (0.0,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class ALbertForMultiLabelClassification(AlbertPreTrainedModel):

    def __init__(self, config:AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            # loss_fct = BCEWithLogLoss()
            # loss = loss_fct(logits, labels)
            # loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        else:
            outputs = (0.0,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class ElectraForMultiLabelClassification(ElectraPreTrainedModel):

    def __init__(self, config:ElectraConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,)

        if labels is not None:
            loss_fct = BCEWithLogLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs
        else:
            outputs = (0.0,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
