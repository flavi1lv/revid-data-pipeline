let currentDataset = [];
let filteredDataset = [];
let currentPage = 1;
const ITEMS_PER_PAGE = 5;
let logSource = null;

// Variables pour l'enregistrement micro
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let micTimeout = null;

function startLogStream() {
    if (logSource) logSource.close();
    logSource = new EventSource('/api/logs');
    logSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.log) appendLog(data.log);
        if (data.done) { logSource.close(); logSource = null; }
    };
}

function appendLog(text) {
    const terminal = document.getElementById('logTerminal');
    const div = document.createElement('div');
    if (text.includes('✅') || text.includes('🎉') || text.includes('✨') || text.includes('🏆')) div.className = 'log-success';
    else if (text.includes('❌') || text.includes('Erreur')) div.className = 'log-error';
    else if (text.includes('⚠️') || text.includes('⏳') || text.includes('📌') || text.includes('🏋️‍♂️')) div.className = 'log-warn';
    div.textContent = text;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
}

function setBadge(id, state, label) {
    const badge = document.getElementById(id);
    if(badge) {
        badge.className = `status-badge status-${state}`;
        badge.textContent = label;
    }
}

function pollStatus() {
    setInterval(async () => {
        try {
            const r = await fetch('/api/status');
            const s = await r.json();
            
            if (s.type === 'generate') {
                if (s.running) setBadge('genBadge', 'running', 'EN COURS');
                else if (s.result === 'success') {
                    if (document.getElementById('genBadge').textContent !== 'TERMINÉ') {
                        setBadge('genBadge', 'done', 'TERMINÉ');
                        loadDataset(); 
                    }
                }
                else if (s.result === 'error') setBadge('genBadge', 'error', 'ERREUR');
            }
            
            if (s.type === 'train') {
                if (s.running) setBadge('trainBadge', 'running', 'EN COURS');
                else if (s.result === 'success') setBadge('trainBadge', 'done', 'TERMINÉ');
                else if (s.result === 'error') setBadge('trainBadge', 'error', 'ERREUR');
            }
        } catch(e) {}
    }, 3000);
}

async function loadDataset() {
    const r = await fetch('/api/dataset');
    currentDataset = await r.json();
    const total = currentDataset.reduce((sum, v) => sum + v.nb_images, 0);
    document.getElementById('globalStats').innerHTML =
        `Total : <b>${currentDataset.length} véhicules</b> | <b>${total} images</b>`;
    handleSearch();
}

function handleSearch() {
    const term = document.getElementById('searchInput').value.toLowerCase();
    filteredDataset = currentDataset.filter(v => v.nom_propre.toLowerCase().includes(term));
    currentPage = 1;
    renderPage();
}

function changePage(delta) { currentPage += delta; renderPage(); }

function renderPage() {
    const listDiv = document.getElementById('datasetList');
    const paginationDiv = document.getElementById('paginationControls');

    if (!filteredDataset.length) {
        listDiv.innerHTML = "<p style='color: var(--text-muted); text-align:center; margin-top:20px;'>Aucun véhicule trouvé.</p>";
        paginationDiv.style.display = "none";
        return;
    }

    const totalPages = Math.ceil(filteredDataset.length / ITEMS_PER_PAGE);
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    const items = filteredDataset.slice(startIndex, startIndex + ITEMS_PER_PAGE);

    listDiv.innerHTML = items.map(v => `
        <div class="vehicle-card">
            <div class="vehicle-info">
                <strong>${v.nom_propre}</strong>
                <span>${v.nb_images} images extraites</span>
            </div>
            <button class="btn-danger" onclick="deleteVehicle('${v.nom_dossier}')">Supprimer</button>
        </div>`
    ).join('');

    if (totalPages > 1) {
        paginationDiv.style.display = "flex";
        document.getElementById('pageInfo').textContent = `Page ${currentPage} / ${totalPages}`;
        document.getElementById('btnPrev').disabled = currentPage === 1;
        document.getElementById('btnNext').disabled = currentPage === totalPages;
    } else {
        paginationDiv.style.display = "none";
    }
}

async function startPipeline() {
    const vehicle = document.getElementById('vehicleInput').value.trim();
    const minutes = parseInt(document.getElementById('targetMinutes').value);

    if (!vehicle) return alert("Veuillez entrer un modèle de véhicule.");
    if (minutes < 1) return alert("Minimum 1 minute.");

    document.getElementById('logTerminal').innerHTML = '';
    appendLog(`🚀 Démarrage du pipeline pour : ${vehicle} (${minutes} min)`);

    document.getElementById('startButton').disabled = true;
    setBadge('genBadge', 'running', 'EN COURS');

    startLogStream();

    const r = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vehicle, target_minutes: minutes })
    });
    const data = await r.json();

    if (data.status === 'error') {
        appendLog(`❌ ${data.message}`);
        setBadge('genBadge', 'error', 'ERREUR');
    }
    document.getElementById('startButton').disabled = false;
    document.getElementById('vehicleInput').value = '';
}

async function deleteVehicle(folderName) {
    if (!confirm(`Supprimer définitivement ${folderName.replace(/_/g, ' ')} ?`)) return;
    await fetch('/api/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: folderName })
    });
    loadDataset();
}

async function startTraining() {
    if (currentDataset.length < 2)
        return alert("Il faut au moins 2 véhicules pour entraîner !");

    document.getElementById('logTerminal').innerHTML = '';
    document.getElementById('trainButton').disabled = true;
    setBadge('trainBadge', 'running', 'EN COURS');
    document.getElementById('trainStatus').textContent = "⏳ Entraînement en cours... Suivi des époques ci-dessous.";

    startLogStream();

    const r = await fetch('/api/train', { method: 'POST' });
    const data = await r.json();

    if (data.status === 'success') {
        setBadge('trainBadge', 'done', 'TERMINÉ');
        document.getElementById('trainStatus').textContent = "✅ Modèle sauvegardé dans /models/";
    } else {
        setBadge('trainBadge', 'error', 'ERREUR');
        document.getElementById('trainStatus').textContent = `❌ ${data.message}`;
    }
    document.getElementById('trainButton').disabled = false;
}

// ── LOGIQUE COMMUNE D'ENVOI À L'IA ──
async function sendAudioToIA(audioBlob, filename) {
    const resultDiv = document.getElementById('iaResult');
    resultDiv.style.display = "block";
    resultDiv.innerHTML = "⏳ Analyse et conversion en cours...";

    const formData = new FormData();
    formData.append("file", audioBlob, filename);

    try {
        const r = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await r.json();

        if (data.error) {
            resultDiv.innerHTML = `❌ Erreur : ${data.error}`;
        } else {
            resultDiv.innerHTML = `🎯 Détection : <span style="color:var(--primary)">${data.vehicule}</span> <br><span style="font-size:14px; color:var(--text-muted)">Confiance : ${data.confiance}%</span>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `❌ Erreur de connexion avec le serveur IA.`;
    }
}

// TÉLÉVERSEMENT PAR FICHIER AUDIO
function testIAWithFile() {
    const fileInput = document.getElementById('testAudio');
    if (!fileInput.files[0]) return alert("Veuillez d'abord sélectionner un fichier audio.");
    sendAudioToIA(fileInput.files[0], fileInput.files[0].name);
}

// ENREGISTREMENT MICRO EN DIRECT
async function toggleMicRecording() {
    const micButton = document.getElementById('micButton');
    
    if (!isRecording) {
        // Démarrer l'enregistrement
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToIA(audioBlob, "recording.wav");
                
                // Couper le micro physiquement
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            micButton.className = "recording";
            micButton.textContent = "⏹️ Arrêter l'enregistrement (0s...)";
            
            // 10 secondes (pour caler avec la limite librosa)
            let elapsed = 0;
            micTimeout = setInterval(() => {
                elapsed++;
                micButton.textContent = `⏹️ Arrêter l'enregistrement (${elapsed}s / 10s)`;
                if (elapsed >= 10) {
                    toggleMicRecording();
                }
            }, 1000);

        } catch (err) {
            alert("Impossible d'accéder au micro. Vérifiez les autorisations de votre navigateur.");
            console.error(err);
        }
    } else {
        // Arrêter l'enregistrement
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
        clearInterval(micTimeout);
        isRecording = false;
        micButton.className = "btn-warn";
        micButton.textContent = "🎙️ Enregistrer le micro (max 10s)";
    }
}

loadDataset();
pollStatus();