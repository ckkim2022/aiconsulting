
import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import re

words = {
    "adjective": [
        "unhappy", "stressed", "confused", "worried", "anxious",
        "frustrated", "discontented", "hopeless", "irate", "melancholic"
    ],
    "activity": [
        "be alone", "escape", "cry", "scream", "sleep",
        "hide", "relax", "reflect", "read", "meditate"
    ],
    "topic": [
        "problems", "feelings", "thoughts", "confusion", "life",
        "challenges", "decisions", "dilemmas", "responsibilities", "opportunities"
    ],
    "relation": [
        "friend", "boyfriend", "girlfriend", "classmate", "best friend",
        "colleague", "partner", "sibling", "mentor", "acquaintance"
    ],
    "emotion": [
        "lonely", "overwhelmed", "excited", "nervous", "sad",
        "joyful", "fearful", "content", "anxious", "hopeful"
    ],
    "subject": [
        "math", "history", "science", "English", "art",
        "economics", "philosophy", "geography", "biology", "physics"
    ],
    "reason": [
        "I am too thin", "I am overweight", "I look different", "people tease me", "I don't fit in",
        "I am too tall", "I have braces", "I dress differently", "I feel out of place", "my accent stands out"
    ]
}


data_templates = {
    "personal": [
        "I feel {adjective} these days",
        "Sometimes I just want to {activity}",
        "I'm not sure who I can talk to about my {topic}",
        "Feeling {adjective} has become a daily struggle for me",
        "I need some time to {activity} because of my {topic}",
        "My days are filled with {emotion} and it's hard to cope",
        "I often worry about my future and feel {adjective}",
        "Trying to manage my emotions about {topic} is exhausting",
        "I don't feel like I can share my true feelings with anyone",
        "The thought of tomorrow makes me feel {adjective}",
        "My {topic} are overwhelming, and I need support",
        "It's a challenge to stay positive when I feel {adjective}",
        "At times, I feel {adjective} without any apparent reason",
        "I struggle to find joy in things I once loved",
        "I'm searching for ways to deal with my {emotion} state",
        "Sometimes, it feels like my {topic} control my life",
        "I wish I could talk to someone about feeling {adjective}",
        "Every morning brings a sense of {emotion} that's hard to shake off",
        "Finding peace has become my daily goal amidst {emotion} days"
    ],
    "academic performance": [
        "I'm really worried about my upcoming {subject} exam",
        "Lately, my grades in {subject} have been {adjective}",
        "Studying {subject} has become increasingly {adjective}",
        "I'm concerned my {subject} skills aren't improving",
        "The thought of failing {subject} makes me feel {adjective}",
        "My {subject} professor suggested I need extra help",
        "I feel like I'm falling behind in {subject}",
        "No matter how much I study, {subject} seems impossible",
        "I'm scared I won't graduate because of my {subject} grades",
        "Every {subject} class brings a sense of dread",
        "I'm not grasping the concepts in {subject} at all",
        "My performance in {subject} is not what I hoped it would be",
        "Fearing the next {subject} test keeps me up at night",
        "I get anxious just thinking about my {subject} homework",
        "Group projects in {subject} make me feel {adjective}",
        "I'm struggling to keep up with the {subject} coursework",
        "The workload for {subject} is more than I can handle",
        "Exams in {subject} always feel like an impossible challenge",
        "Discussions in {subject} class often leave me feeling {adjective}"
    ],
    "appearance": [
        "I feel {adjective} about the way I look",
        "What can I do to look more {adjective}?",
        "Feeling {adjective} when I see my reflection is tough",
        "Changing my style has made me feel more {adjective}",
        "I am considering changing my look to feel {adjective}",
        "Dealing with comments about my appearance makes me {emotion}",
        "Sometimes I avoid mirrors because I feel {adjective}",
        "I wish I felt more {emotion} about my body image",
        "People's remarks about how I look make me feel {adjective}",
        "Trying new looks has not helped me feel less {adjective}",
        "I struggle with self-acceptance because I feel {adjective}",
        "Feeling {adjective} about my body is a daily issue",
        "My appearance doesn't reflect how I feel inside",
        "I'm experimenting with my style to boost my {emotion}",
        "Others' opinions on my looks are often too {adjective}",
        "I've started to feel {adjective} about going out in public",
        "Personal style has become a source of stress for me",
        "Society's beauty standards make me feel {adjective}",
        "I'm learning to embrace my looks and feel {emotion}"
    ],
    "relationship": [
        "I had a fight with my {relation}",
        "I feel {emotion} when I'm around my friends",
        "It's not easy to maintain a good relationship with my {relation}",
        "Having {relation} issues has made me very {adjective}",
        "I rely on my {relation} for emotional support",
        "Misunderstandings with my {relation} are frequent and make me feel {adjective}",
        "I'm considering ending my relationship with my {relation}",
        "Feeling disconnected from my {relation} has been hard",
        "I need advice on how to handle conflicts with my {relation}",
        "Trust issues with my {relation} are causing me stress",
        "Communication gaps with my {relation} are becoming common",
        "It's hard to express my feelings to my {relation}",
        "Sometimes I doubt the future of my relationship with {relation}",
        "We used to be close, but now I feel {emotion} around my {relation}",
        "It's been difficult to forgive my {relation} after what happened",
        "Our relationship has been very {adjective} lately",
        "I often feel {adjective} when I think about my relationship with {relation}",
        "Rebuilding trust with my {relation} seems daunting",
        "My {relation} and I are working on better understanding each other"
    ],
    "etc": [
        "I'm not sure how to balance school and my hobbies",
        "I'm feeling {emotion} about my future",
        "My schedule is becoming a burden, and I'm always {adjective}",
        "Finding a good balance between work and leisure is challenging",
        "The uncertainty of what comes next makes me {adjective}",
        "I wish I had someone to talk to about my {topic}",
        "I feel {adjective} about the choices I've made",
        "It's hard to stay motivated with so much uncertainty",
        "My plans for the future are causing me stress",
        "I'm reconsidering my career path because I feel {adjective}",
        "Making decisions about my life is increasingly difficult",
        "I'm unsure how to approach my long-term goals",
        "Life's unpredictability is making me feel very {adjective}",
        "I'm struggling to find my purpose and feel {emotion}",
        "Adjusting to new changes has been {adjective}",
        "Exploring different paths has left me feeling {adjective}",
        "The pressure to succeed is overwhelming and makes me feel {adjective}",
        "I often ponder the direction my life is taking",
        "Finding a career that fulfills me is becoming increasingly important"
    ]
}


label_dict = {
    "personal": 0,
    "academic performance": 1,
    "appearance": 2,
    "relationship": 3,
    "etc": 4
}

# def fill_template(template, word_choices):
#     for key, value in word_choices.items():
#         template = template.replace("{" + key + "}", value)
#     return template

# def generate_data(data_templates, words, label_dict):
#     data = []
#     for category, templates in data_templates.items():
#         label = label_dict[category]
#         for template in templates:
#             # Extract placeholders using regex to match text within curly braces
#             keys = re.findall(r'\{(.*?)\}', template)
#             # Generate all combinations for the placeholders
#             combinations = itertools.product(*(words[key] for key in keys))
#             for combination in combinations:
#                 word_choices = dict(zip(keys, combination))
#                 filled_template = fill_template(template, word_choices)
#                 data.append((filled_template, label))
#     return pd.DataFrame(data, columns=["text", "label"])

# df = generate_data(data_templates, words, label_dict)
# df.to_csv("sample_data.csv")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))

# tokenized_inputs = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=512)

# class TextDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

# dataset = TextDataset(tokenized_inputs, df['label'].tolist())

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=5,
#     per_device_train_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset
# )

# trainer.train()
# model.save_pretrained("./saved_model")
# tokenizer.save_pretrained("./saved_model")

import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def predict_intent(text,tokenizer,model):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return list(label_dict.keys())[predictions.item()]

def debug_tokenization(text,tokenizer):
    text = preprocess_text(text)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
solutions = {
    "personal": [
        "Consider setting aside some time each day for relaxation and self-reflection.",
        "It might help to talk to someone you trust about your feelings.",
        "Keeping a journal can also be a great way to express what you're feeling inside."
    ],
    "academic performance": [
        "Organizing study time and breaking tasks into smaller chunks could improve your focus.",
        "Don't hesitate to ask for help from teachers or consider joining a study group.",
        "Make sure you're taking regular breaks to rest your mind."
    ],
    "appearance": [
        "Remember that everyone's unique, and embracing your individuality is key.",
        "Talking to someone about how you feel can often put things into perspective.",
        "Maintaining healthy eating and exercise habits are important for both your physical and mental well-being."
    ],
    "relationship": [
        "Effective communication is crucial; try to express your feelings openly and honestly.",
        "Listening is just as important as talking; make sure you understand others' perspectives.",
        "If conflicts are frequent, it might help to involve a mediator or counselor."
    ],
    "etc": [
        "Sometimes taking a step back to reassess our priorities can provide clarity.",
        "Seeking advice from those who have experienced similar situations can be helpful.",
        "If you feel overwhelmed, professional help such as counseling can provide significant support."
    ]
}


def get_response(user_input):
    model_path = "./saved_model"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    debug_tokenization(user_input,tokenizer)
    predicted_category = predict_intent(user_input,tokenizer,model)
    response = random.choice(solutions[predicted_category])
    return response


