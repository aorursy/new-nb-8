import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.append(package_dir)

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import torch.utils.data

import numpy as np

import pandas as pd

from tqdm import tqdm

import os

import warnings

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

from pytorch_pretrained_bert import BertConfig

import gc



warnings.filterwarnings(action='once')

device = torch.device('cuda')



def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)



MAX_SEQUENCE_LENGTH = 220

SEED = 1234

BATCH_SIZE = 32

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

LARGE_BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

# Pretrained BERT models - Google's pretrained BERT model

BERT_SMALL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

BERT_LARGE_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'



# JIGSAW fine-tuned BERT models

JIGSAW_BERT_SMALL_MODEL_PATH = '../input/bert-inference/bert/bert_pytorch.bin'

JIGSAW_BERT_LARGE_MODEL_PATH = '../input/jigsawpretrainedbertmodels/jigsaw-bert-large-uncased-len-220-fp16/epoch-1/pytorch_model.bin'

JIGSAW_BERT_SMALL_JSON_PATH = '../input/bert-inference/bert/bert_config.json'

JIGSAW_BERT_LARGE_JSON_PATH = '../input/jigsawpretrainedbertmodels/jigsaw-bert-large-uncased-len-220-fp16/epoch-1/config.json'

NUM_BERT_MODELS = 2

INFER_BATCH_SIZE = 64



train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



test_preds = np.zeros((test_df.shape[0],NUM_BERT_MODELS))

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True







print("Predicting BERT large model......")



# Prepare data

bert_config = BertConfig(JIGSAW_BERT_LARGE_JSON_PATH)

tokenizer = BertTokenizer.from_pretrained(BERT_LARGE_PATH, cache_dir=None,do_lower_case=True)

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))      



# Load fine-tuned BERT model

gc.collect()

model = BertForSequenceClassification(bert_config, num_labels=1)

model.load_state_dict(torch.load(JIGSAW_BERT_LARGE_MODEL_PATH))

model.to(device)

for param in model.parameters():

    param.requires_grad = False

model.eval()



# Predicting

model_preds = np.zeros((len(X_test)))

test_loader = torch.utils.data.DataLoader(test, batch_size=INFER_BATCH_SIZE, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

        model_preds[i * INFER_BATCH_SIZE:(i + 1) * INFER_BATCH_SIZE] = pred[:, 0].detach().cpu().squeeze().numpy()



test_preds[:,0] = torch.sigmoid(torch.tensor(model_preds)).numpy().ravel()

del model

gc.collect()



print("Predicting BERT small model......")

bert_config = BertConfig(JIGSAW_BERT_SMALL_JSON_PATH)

tokenizer = BertTokenizer.from_pretrained(BERT_SMALL_PATH, cache_dir=None,do_lower_case=True)

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))      



# # # Load fine-tuned BERT model

model = BertForSequenceClassification(bert_config, num_labels=1)

model.load_state_dict(torch.load(JIGSAW_BERT_SMALL_MODEL_PATH))

model.to(device)

for param in model.parameters():

    param.requires_grad = False

model.eval()



# Predicting

model_preds = np.zeros((len(X_test)))

test_loader = torch.utils.data.DataLoader(test, batch_size=INFER_BATCH_SIZE, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

        model_preds[i * INFER_BATCH_SIZE:(i + 1) * INFER_BATCH_SIZE] = pred[:, 0].detach().cpu().squeeze().numpy()



test_preds[:,1] = torch.sigmoid(torch.tensor(model_preds)).numpy().ravel()



del model

gc.collect()



# Sub-model prediction

bert_submission = pd.DataFrame.from_dict({

'id': test_df['id'],

'prediction': test_preds.mean(axis=1)})

bert_submission.to_csv('bert_submission.csv', index=False)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from keras.preprocessing import text, sequence

from keras import backend as K

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import pickle

tqdm.pandas()
EMBEDDING_PATHS = ['../input/pickled-word-embedding/crawl-300d-2M.pkl',

                 '../input/pickled-word-embedding/glove.840B.300d.pkl']





NUM_MODELS = 2 # The number of classifiers we want to train 

BATCH_SIZE = 512 # can be tuned

LSTM_UNITS = 128 # can be tuned

DENSE_HIDDEN_UNITS = 4*LSTM_UNITS # can betuned

EPOCHS = 4 # The number of epoches we want to train for each classifier

MAX_LEN = 220 # can ben tuned





IDENTITY_COLUMNS = [

    'transgender', 'female', 'homosexual_gay_or_lesbian', 'muslim', 'hindu',

    'white', 'black', 'psychiatric_or_mental_illness', 'jewish'

    ]  



AUX_COLUMNS = ['target', 'severe_toxicity','obscene','identity_attack','insult','threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'
#----------------------------------- Preprocessing-------------------------------------#

SYMBOLS_TO_ISOLATE = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'

SYMBOLS_TO_REMOVE = '\n🍕\r🐵\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'

ISOLATE_DICT = {ord(c):f' {c} ' for c in SYMBOLS_TO_ISOLATE}

REMOVE_DICT = {ord(c):f'' for c in SYMBOLS_TO_REMOVE}

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def handle_punctuation(text):

    text = text.translate(REMOVE_DICT)

    text = text.translate(ISOLATE_DICT)

    return text



def clean_contractions(text, mapping=CONTRACTION_MAPPING):

    '''

    Expand contractions

    '''

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



def preprocess(x):

    x = handle_punctuation(x)

#     x = clean_contractions(x)

    return x



#----------------------------------- Embedding -------------------------------------#

def get_coefs(word, *arr):

    """

    Get word, word_embedding from a pretrained embedding file

    """

    return word, np.asarray(arr,dtype='float32')



def load_embeddings(path):

    if path.split('.')[-1] in ['txt','vec']: # for original pretrained embedding files (extension .text, .vec)

        with open(path,'rb') as f:

            return dict(get_coefs(*line.strip().split(' ')) for line in f)    

    if path.split('.')[-1] =='pkl': # for pickled pretrained embedding files (extention pkl). Loading pickeled embeddings is faster than texts

        with open(path,'rb') as f:

            return pickle.load(f)

    





def build_matrix(word_index, path):

    """

    Here we take each word we've tokenized in our text corpus

    for each word we look up in the pre-trained embedding.

    Each row in this matrix is a corpus word's embedding.

    """

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index)+1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim

        

    

def build_model(embedding_matrix, num_aux_targets):#, loss_weight):

    """

    embedding layer

    droput layer

    2 * bidirectional LSTM layers

    2 * pooling layers

    2 dense layers

    1 softmax layer

    """

    words = Input(shape=(MAX_LEN,)) 

    #Embedding layer takes variable size input

    x = Embedding(*embedding_matrix.shape, weights = [embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    

    #att = Attention(MAX_LEN)(x)

    hidden = concatenate([ 

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x)

        ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result =Dense(num_aux_targets, activation='sigmoid')(hidden)



    model = Model(inputs =words, outputs =[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')

    

    return model
# Preprocess comment texts

train_df['comment_text'] = train_df['comment_text'].progress_apply(lambda x:preprocess(x))

test_df['comment_text'] = test_df['comment_text'].progress_apply(lambda x:preprocess(x))

gc.collect()
x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)



# Convert target probability to 1 or 0 so they can be used for classification

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >=0.5, True, False)
# Return a Keras tokenizer class

tokenizer = text.Tokenizer(filters = CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train)+ list(x_test))

# Turn text to sequences of tokens

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

#Pad sequences to the same length

x_train = sequence.pad_sequences(x_train,maxlen=MAX_LEN)

x_test= sequence.pad_sequences(x_test, maxlen=MAX_LEN)
# Initialize weights

sample_weights = np.ones(len(x_train), dtype=np.float32)

# Add all the values of the identities along rows

sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)

#Add all values of targets*~identity

sample_weights += train_df[TARGET_COLUMN]*(~train_df[IDENTITY_COLUMNS]).sum(axis=1)

#Add all values ~targets*identity

sample_weights += (~train_df[TARGET_COLUMN])*train_df[IDENTITY_COLUMNS].sum(axis=1)

#Normalize them

sample_weights/=sample_weights.mean()
embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index,f) for f in EMBEDDING_PATHS], axis =-1)

print("Embedding matrix shape:", embedding_matrix.shape)

del train_df, tokenizer

gc.collect()
checkpoint_predictions = []

weights = []

NUM_MODELS = 1

for model_idx in range(NUM_MODELS):

    #Passes embedding matrix and aux outputs shape

    model = build_model(embedding_matrix, y_aux_train.shape[-1]) #1/sample_weights.mean())

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=1,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks = [

                LearningRateScheduler(lambda _: 1e-3*(0.55**global_epoch)) # Decayed learning rate

                ]

        )

#         model.save_weights("model_%d_%d.h5" % (model_idx, global_epoch)) # Save model weights

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)

    del model # If a model didn't get deleted Keras will continue training it eventhough build_model() was used to initialize a model

    gc.collect() # It's a good practice to use gc.collect() once the training is done to free up RAM
print (weights)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

lstm_submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

lstm_submission.to_csv('submission.csv', index=False)
submission = pd.DataFrame.from_dict({

'id': test_df['id'],

'prediction': lstm_submission['prediction'].rank(pct=True)*0.4 + bert_submission['prediction'].rank(pct=True)*0.6})

submission.to_csv('submission.csv', index=False)
