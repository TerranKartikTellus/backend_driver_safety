from gtts import gTTS

# The text you want to convert to speech
text = "Hello, how are you today?"

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine, which will generate the speech
speech = gTTS(text=text, lang=language, slow=False)

# Saving the speech as an MP3 file
speech.save("hello.mp3")
