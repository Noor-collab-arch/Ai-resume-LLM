import os
import faiss
import numpy as np
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ["GROQ_API_KEY"] = "paste_your_groq_key"
llm = ChatGroq(model="llama3-8b-8192")  # Use LLaMA3 model for better analysis
embedding_model = HuggingFaceEmbeddings()
def process_pdf(file_path):
    """Extracts text from PDF and returns combined text."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    return " ".join([doc.page_content for doc in split_docs])
def create_faiss_index(texts):
    """Creates FAISS index for storing resume embeddings."""
    vectors = embedding_model.embed_documents(texts)  # Convert text to embeddings
    dimension = len(vectors[0])  # Get embedding size

    faiss_index = faiss.IndexFlatL2(dimension)  # L2 Distance Index
    faiss_index.add(np.array(vectors))  # Add embeddings to FAISS

    return faiss_index, vectors  # Return FAISS index and vector embeddings
def compare_resumes(job_desc_path, resume_paths):
    """Compares resumes with job description and returns results."""
    job_desc_text = process_pdf(job_desc_path)  # Extract job description text
    resume_texts = [process_pdf(resume) for resume in resume_paths]  # Extract resumes text

    # Create FAISS index for resumes
    faiss_index, resume_vectors = create_faiss_index(resume_texts)

    # Embed job description and perform FAISS search
    job_desc_vector = embedding_model.embed_documents([job_desc_text])[0]  # Get embedding for job description
    _, matched_indices = faiss_index.search(np.array([job_desc_vector]), k=len(resume_paths))  # Find best matches

    results = []  # Store results

    for rank, idx in enumerate(matched_indices[0]):  # Iterate over matched resumes
        resume_text = resume_texts[idx]
        resume_name = os.path.basename(resume_paths[idx])

        # Compute Similarity Score (Cosine Similarity)
        similarity = np.dot(job_desc_vector, resume_vectors[idx]) / (
            np.linalg.norm(job_desc_vector) * np.linalg.norm(resume_vectors[idx])
        )
        match_percentage = round(similarity * 100, 2)  # Convert to percentage

        # Query LLM for Detailed Analysis
        response = llm.invoke([
            HumanMessage(content=f"""
            Compare the following resume with the job description.
            Job Description: {job_desc_text}
            Resume: {resume_text}

            Provide the following details:
            1. **generate response in paragraph**
            2. **Key Skills Matched**
            3. **Missing Skills**
            4. **Suggestions for Improvement**

            Format the response in a structured manner.
            """)
        ])

        results.append(f"**🔹 Result for {resume_name}**\n\n**Match Percentage:** {match_percentage}%\n\n" + response.content)

    return "\n\n".join(results)
def resume_match_interface(job_desc_file, resume_files):
    """Handles file inputs from Gradio and returns match results."""
    if not job_desc_file or not resume_files:
        return "**❌ Please upload a job description and at least one resume.**"

    # Convert Gradio file objects to actual file paths
    job_desc_path = job_desc_file.name
    resume_paths = [resume.name for resume in resume_files]

    return compare_resumes(job_desc_path, resume_paths)  # Call main function
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Resume Matching AI Chatbot 🤖")

    with gr.Row():
        job_desc_input = gr.File(label="📄 Upload Job Description (PDF)", type="filepath")
        resume_inputs = gr.File(label="📄 Upload Resumes (PDF)", type="filepath", file_count="multiple")

    output_box = gr.Markdown()

    submit_button = gr.Button("🔍 Analyze Resumes")

    submit_button.click(resume_match_interface, inputs=[job_desc_input, resume_inputs], outputs=output_box)

# Step 🔟: Launch Gradio App
demo.launch()    
