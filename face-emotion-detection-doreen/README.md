# Facial Emotion Recognition

Creator: Doreen Machoni

Recognizing emotions of a face(happy, angry, sad, neutral, surprise, disgust or fear) and outputting the identified emotion in root directory as a snap. Snap output by default outputs the face of interest e.g., fear detected, happy etc.

```python
        if emotion == 'happy':
            cv2.imwrite(f'{emotion}_snapshot.png', frame)
```

## Running the Program

- Assuming you've already downloaded the program and changed directory to it, run the following:

Essential package installation if doesn't exist

```python
pip install -r requirements.txt
```

Running the program

```markdown
python main.py
```
