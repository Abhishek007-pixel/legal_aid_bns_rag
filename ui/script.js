document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const uploadStatus = document.getElementById('uploadStatus');
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const chatHistory = document.getElementById('chatHistory');
    let currentFile = null;

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    async function handleFile(file) {
        if (file.type !== 'application/pdf') {
            uploadStatus.textContent = '❌ Only PDF files are supported';
            uploadStatus.className = 'status-msg status-error';
            return;
        }

        uploadStatus.textContent = '⏳ Uploading & Indexing...';
        uploadStatus.className = 'status-msg';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok) {
                currentFile = data.filename;
                uploadStatus.textContent = `✅ Indexed: ${data.filename} (${data.chunks_added} chunks)`;
                uploadStatus.className = 'status-msg status-success';
                addSystemMessage(`I've studied **${data.filename}**. You can now ask questions about it!`);
            } else {
                throw new Error(data.detail || data.error || 'Upload failed');
            }
        } catch (e) {
            uploadStatus.textContent = `❌ ${e.message}`;
            uploadStatus.className = 'status-msg status-error';
        }
    }

    // Chat
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (!q) return;

        addMessage(q, 'user');
        questionInput.value = '';

        // Show loading
        const loadingId = addMessage('Thinking...', 'bot', true);

        try {
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: q,
                    filter_filename: currentFile
                })
            });
            const data = await res.json();

            removeMessage(loadingId);

            if (data.fallback) {
                addMessage(data.answer, 'bot', false, true); // Warning style
            } else {
                addMessage(data.answer, 'bot');
            }

            if (data.citations && data.citations.length) {
                addCitations(data.citations);
            }

        } catch (e) {
            removeMessage(loadingId);
            addMessage(`Error: ${e.message}`, 'bot');
        }
    });

    function addMessage(text, sender, isLoading = false, isWarning = false) {
        const div = document.createElement('div');
        div.className = `message ${sender}-message ${isLoading ? 'loading' : ''} ${isWarning ? 'warning' : ''}`;
        div.id = isLoading ? 'loading-' + Date.now() : '';

        // Simple markdown formatting
        if (!isLoading) {
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/\n/g, '<br>');
        }

        div.innerHTML = text;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return div.id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'message bot-message system';
        div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function addCitations(citations) {
        const div = document.createElement('div');
        div.className = 'citations';
        div.innerHTML = '<h4>Sources:</h4>';

        citations.slice(0, 3).forEach(c => { // Limit to 3
            const item = document.createElement('div');
            item.className = 'citation-item';
            const title = c.title || c.source || 'Unknown';
            const score = c.score ? `(Score: ${c.score.toFixed(2)})` : '';
            item.textContent = `${title} ${score}`;
            div.appendChild(item);
        });

        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
