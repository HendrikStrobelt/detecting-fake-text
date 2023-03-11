from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, BertForMaskedLM)


def run(model_name_or_path):

    GPT2Tokenizer.from_pretrained(model_name_or_path)
    GPT2LMHeadModel.from_pretrained(model_name_or_path)
    print("Loaded GPT-2 model!")


if __name__ == '__main__':
    run("gpt2")

