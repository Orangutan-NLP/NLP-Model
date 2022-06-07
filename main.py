import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


parser = argparse.ArgumentParser(description='Orangutan-NLP fine-tuned on KoGPT-2')
parser.add_argument('--chat', action='store_true', default=False, help='Filter received chat data')
parser.add_argument('--model_params', type=str, default='model_chp/orangutan.ckpt')
parser.add_argument('--train', action='store_true', default=False, help='Start fine tuning')


U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
                                                    bos_token=BOS, 
                                                    eos_token=EOS, 
                                                    unk_token='<unk>', 
                                                    pad_token=PAD, 
                                                    mask_token=MASK) 


class OrangutanData(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        # self.first = True
        self.o_token = U_TKN
        self.f_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        chat = self._data.iloc[idx]
        o = chat['original']
        f = chat['filtered']
        label = str(chat['label'])
        o_toked = self.tokenizer.tokenize(self.o_token + o + self.sent_token + label)   
        o_len = len(o_toked)
        f_toked = self.tokenizer.tokenize(self.f_token + f + self.eos)
        f_len = len(f_toked)
        
        if o_len + f_len > self.max_len:
            f_len = self.max_len - o_len
            if f_len <= 0:
                o_toked = o_toked[-(int(self.max_len/2)):]
                o_len = len(o_toked)
                f_len = self.max_len - o_len
                assert f_len > 0
            f_toked = f_toked[:f_len]
            f_len = len(f_toked)
            assert f_len == len(f_toked), f'{f_len} ==? {len(f_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [self.mask] * o_len + f_toked[1:]

        mask = [0] * o_len + [1] * f_len + [0] * (self.max_len - o_len - f_len)

        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(o_toked + f_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return (token_ids, np.array(mask), labels_ids)


class KoGPT2Finetuning(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Finetuning, self).__init__()
        self.hp = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len', type=int, default=32, help='max sentence length on input (default: 32)')
        parser.add_argument('--batch-size', type=int, default=96, help='batch size for training (default: 96)')
        parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hp.lr, 
                          correct_bias=False)
        # Warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hp.max_epochs
        num_warmup_steps = int(num_train_steps * self.hp.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'name': 'cosine_schedule_with_warmup', 
                        'monitor': 'loss', 
                        'interval': 'step', 
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('/home/jihye/jihye/Orangutan-NLP/Data/orangutan_data_label.csv', sep='\t')
        self.train_set = OrangutanData(data, max_len=self.hp.max_len)
        train_dataloader = DataLoader(self.train_set, 
                                      batch_size=self.hp.batch_size, 
                                      num_workers=16, 
                                      shuffle=True, 
                                      collate_fn=self._collate_fn)
        return train_dataloader

    def filterchatting(self, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            while 1:
                q = input('User input > ').strip()
                if q == 'quit':
                    break
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                print("Filtered chat > {}".format(a.strip()))


parser = KoGPT2Finetuning.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


if __name__ == "__main__":
    if args.train:
        # 사전훈련된 KoGPT2를 Orangutan-NLP 데이터로 파인튜닝
        # !CUDA_VISIBLE_DEVICES=0 python3 main.py --train --gpus 1 --max_epochs 10
        
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
        )
        model = KoGPT2Finetuning(args)
        model.train()
        trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        
    if args.chat:
        # 대화 테스트, `quit`를 입력하면 대화 종료
        # !CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 1 --chat
        
        model = KoGPT2Finetuning.load_from_checkpoint(checkpoint_path=args.model_params, hparams=args)
        model.filterchatting()