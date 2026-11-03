import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag import add_pdf, ask

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    name = secure_filename(file.filename)
    path = os.path.join("uploads", name)
    file.save(path)

    count = add_pdf(path, name)
    if count == 0:
        return jsonify({"error": "No text found in this PDF"}), 400
    return jsonify({"message": f"Done! {count} chunks saved."})


@app.route("/ask", methods=["POST"])
def ask_question():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Please type a question"}), 400
    answer, pages = ask(question)
    return jsonify({"answer": answer, "pages": pages})


if __name__ == "__main__":
    app.run(debug=True)