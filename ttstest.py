from gtts import gTTS
from playsound import playsound

print('Starting basic gtts working...')
print("Converting your text to sound . . .")

tts = gTTS(text="Fire is detected, please come out of the place.", lang='en')
tts.save("voice.mp3")
print("Starting audio. . .")
playsound('voice.mp3')
print("Thank You!!!")