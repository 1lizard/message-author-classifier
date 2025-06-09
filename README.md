# Message Author Classifier

This is a little project where I trained a neural network to guess who wrote a message in a chat — me or someone else.

It uses a simple bag-of-words setup and a PyTorch model trained on 5000 labeled messages.

---

## 🛠 How it works

- Each message gets turned into a vector based on word counts
- The model is a basic feedforward neural net
- You can type your own messages after training and see if it guesses who wrote them

---

## 📈 Results

At first, training accuracy was around 75% and test accuracy around 68%.  
After cleaning the data, balancing classes, and tweaking the model a bit, I got:

- **~82% training accuracy**
- **~70% test accuracy**

I initially tested 2 layers but after trying dropout, and other tweaks, just one layer worked just as well.

---

## 🤷 What I learned

- Two layers weren’t needed — didn’t improve anything
- The data and features (just word counts) are probably too limited to go much further
- The difference in message styles in the chat I tested wasn't sufficient to make predictions more accurate (as a human I wouldn't be able to tell that much more accurately without knowing the context)
- Shows much higher accuracy when manually tested on longer messages

---

## 🧪 Try it out

After training, you can type a message and it’ll guess if it was written by one or the other person.

---

## Files

- `train.py` — training script
- `data.py` - json file processing
- `README.md` — this file
