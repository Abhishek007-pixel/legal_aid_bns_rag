\# ⚖️ LegalAid · Bharatiya Nyaya Sanhita (BNS) RAG Assistant



A \*\*Retrieval-Augmented Generation (RAG)\*\* powered Streamlit app that provides section-wise explanations of the \*\*Bharatiya Nyaya Sanhita (BNS)\*\* — the Indian Penal Code replacement under the new criminal law framework.



The app allows users to ask legal questions (e.g., \*"What does Section 303 state?"\*) and returns grounded answers from the \*\*BNS PDF\*\* using embeddings and context-aware reasoning.



---



\## 🧠 Features



\- 🔍 \*\*Section-wise Legal Search\*\* – Query any section of the Bharatiya Nyaya Sanhita.

\- 📚 \*\*Context-Aware Answers\*\* – Retrieves and summarizes relevant law text.

\- ⚙️ \*\*RAG Pipeline\*\* – Combines document embeddings with LLM reasoning.

\- 🌐 \*\*Streamlit Interface\*\* – Clean, responsive UI for public access.

\- 🧾 \*\*Evidence-Based Output\*\* – Each answer includes cited references and context sources.



---



\## 🏗️ Project Architecture



📦 legal\_aid

├── data/

│ └── raw/BNS\_Part1.pdf

├── scripts/

│ ├── ingest.py # Ingests \& chunks the BNS PDF

│ ├── eval.py # Evaluates retrieval accuracy

│ ├── debug\_search.py # Tests semantic search

│ └── list\_models.py # Lists available embedding models

├── ui/

│ └── app.py # Streamlit frontend

├── requirements.txt

└── README.md



yaml





---



\## 🧰 Tech Stack



| Component | Technology |

|------------|-------------|

| Framework | Streamlit |

| Language | Python |

| Model | OpenAI GPT / Hugging Face RAG |

| Embeddings | Nomic / Instructor / OpenAI |

| Database | FAISS / Chroma |

| Document Source | Bharatiya Nyaya Sanhita (BNS) PDF |



---



\## 🖼️ Screenshots



| Home Screen | Chat Interface |

|--------------|----------------|

| !\[Home](assets/home.png) | !\[Chat](assets/chat.png) |



\*(Place your images in `assets/` folder and replace names accordingly.)\*



---



\## 🎥 Demo Video



🎬 \*\*\[Click to Watch Demo](https://youtu.be/m9klEMLh5MU)\*\*  

\*(Replace the above link once you upload your project walkthrough.)\*



---



\## ⚙️ Installation \& Setup



1️⃣ \*\*Clone the repository\*\*

```bash

git clone https://github.com/<your-username>/legal\_aid.git

cd legal\_aid

2️⃣ Create a virtual environment



bash



python -m venv .venv

source .venv/Scripts/activate     # (Windows)

3️⃣ Install dependencies



bash



pip install -r requirements.txt

4️⃣ Run the app



bash



streamlit run ui/app.py

🧩 Example Queries

“What does Section 271 of BNS state?”



“Define theft under the Bharatiya Nyaya Sanhita.”



“what is defination of rape and its punishment in BNS.”



“what are all sections in BNS.”



📈 Future Improvements

🔹 Support for Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Sakshya Adhiniyam (BSA)



🔹 Enhanced summarization and cross-section linking



🔹 Voice query integration



🔹 Deployment on Hugging Face / Streamlit Cloud



👨‍💻 Author

Ankur Gupta

📍 India

🔗 GitHub | LinkedIn



⚠️ Disclaimer

This app provides general legal information based on the Bharatiya Nyaya Sanhita.

It is not legal advice. For specific legal issues, consult a qualified advocate.



⭐ Contribute

If you like this project:



Give it a ⭐ on GitHub



Report issues or suggest improvements via Pull Requests



yaml





---



Would you like me to:

1\. ✅ Save this content into a `README.md` file in your repo automatically (so you can push it)?  

2\. Or 📝 would you like to review/edit a few sections first (like video link, image names, and author links)?

