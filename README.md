# NLU-Enhanced RAG Chatbot with Voice & Text Capabilities

## 🌟 Overview

This project implements a sophisticated conversational AI system that combines Retrieval-Augmented Generation (RAG), Natural Language Understanding (NLU), and voice interaction capabilities. The chatbot can process both voice and text inputs, understand user intent, extract entities, and provide responses in both text and speech formats.

## 🚀 Key Features

- Voice-to-Text transcription using OpenAI's Whisper
- Text-to-Speech conversion using gTTS
- Natural Language Understanding with Named Entity Recognition
- Intent Classification
- RAG (Retrieval Augmented Generation) implementation
- Conversational memory management
- Multi-modal input/output (voice and text)
- Web-based user interface using Gradio

## 🛠️ Technology Stack

### Core AI/ML Components
- **Whisper**: OpenAI's speech recognition model for voice transcription
- **BERT**: Fine-tuned BERT model for Named Entity Recognition (NER)
- **BART**: Facebook's BART model for intent classification
- **LLaMA**: Large Language Model for text generation
- **GPT4All**: For generating embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector storage and retrieval

### Framework & Libraries
- **LangChain**: For RAG implementation and conversation management
- **Ollama**: Local LLM hosting and inference
- **Gradio**: Web interface development
- **Transformers**: Hugging Face's library for NLP tasks
- **gTTS**: Google Text-to-Speech for audio response generation

### Data Management
- **FAISS Vector Store**: For storing and retrieving document embeddings
- **ConversationBufferMemory**: For maintaining conversation history

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama server running locally
- Sufficient storage for model weights
- CUDA-compatible GPU (recommended for optimal performance)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Bhudil/Voice_ChatBot.git
cd nlu-enhanced-rag-chatbot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up the knowledge base directory:
```bash
mkdir -p knowledge-base
```

4. Start the Ollama server:
```bash
ollama serve
```

## 📁 Project Structure

```
.
├── knowledge-base/          # Directory for storing document files
├── final.py                 # Main application code
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 💻 Usage

1. Start the application:
```bash
python final.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. You can interact with the chatbot in two ways:
   - Speak into your microphone using the audio input
   - Type your query in the text input field

4. The system will provide:
   - Text response
   - Audio response
   - Detected intent
   - Extracted entities

## 🔄 How It Works

1. **Input Processing**:
   - Voice input is transcribed using Whisper
   - Text input is processed directly

2. **NLU Processing**:
   - Intent classification using BART
   - Named Entity Recognition using BERT

3. **RAG Pipeline**:
   - Query is processed against the knowledge base
   - Relevant documents are retrieved using FAISS
   - LLaMA generates contextual responses

4. **Response Generation**:
   - Text response is generated
   - Audio response is synthesized using gTTS

## ⚙️ Configuration

The system uses several configurable parameters:

- `MODEL`: LLM model name (default: "llama3.2")
- `chunk_size`: Document splitting size (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- Knowledge base location: "knowledge-base/*"

## Snippets

![Screenshot (224)](https://github.com/user-attachments/assets/5361d7d2-3ca8-436a-85cc-b45f7cdce488)


Note that if the query is irrelevant to the context of the said company which the customer wants to analyse, the bot will not respond accordingly

![Screenshot (223)](https://github.com/user-attachments/assets/211bfb92-b467-4c86-a6fd-82b8e12d5ec0)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

- Ensure your knowledge base documents are in Markdown (.md) format
- The system requires an active internet connection for text-to-speech conversion
- GPU acceleration is recommended for optimal performance
- Make sure to comply with all model licensing requirements

## 🐛 Troubleshooting

Common issues and solutions:

1. **Audio device not found**: Check system microphone permissions
2. **Ollama connection error**: Ensure Ollama server is running on port 11434
3. **Memory issues**: Reduce chunk size or use smaller models
4. **Slow response time**: Consider using GPU acceleration or smaller models

## 📚 References

- [Whisper Documentation](https://github.com/openai/whisper)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [Gradio Documentation](https://gradio.app/docs/)
- [Ollama Documentation](https://ollama.ai/docs)
