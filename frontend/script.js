//JS will use came case for variable and function names


// Configuration
const CONFIG = {
    noteHeight: 20,
    beatWidth: 40,
    totalBeats: 32, // 8 bars of 4/4
    startOctave: 0, // A0 is in octave 0
    endOctave: 8,   // C8 is in octave 8
    tempo: 120,
    snap: 4 // 1/4 beat (16th note)
};

// Project State
let project = {
    tracks: [
        { id: 0, name: "Piano", instrument: "piano", color: "#4CAF50", notes: [] },
        { id: 1, name: "Drums", instrument: "drums", color: "#FF9800", notes: [] },
        { id: 2, name: "Bass", instrument: "bass", color: "#2196F3", notes: [] },
        { id: 3, name: "Strings", instrument: "strings", color: "#9C27B0", notes: [] }
    ],
    activeTrackId: 0
};

// History for Undo/Redo
let history = [];
let historyIndex = -1;

function saveState() {
    const state = JSON.parse(JSON.stringify(project));
    
    // Remove future history if we are in the middle
    if (historyIndex < history.length - 1) {
        history = history.slice(0, historyIndex + 1);
    }
    
    history.push(state);
    historyIndex++;
    
    // Limit history size
    if (history.length > 50) {
        history.shift();
        historyIndex--;
    }
}

function undo() {
    if (historyIndex > 0) {
        historyIndex--;
        project = JSON.parse(JSON.stringify(history[historyIndex]));
        render();
    }
}

function redo() {
    if (historyIndex < history.length - 1) {
        historyIndex++;
        project = JSON.parse(JSON.stringify(history[historyIndex]));
        render();
    }
}

// Initialize History
saveState();

let isPlaying = false;

// HTML Elements
const canvas = document.getElementById('pianoRollCanvas');
const ctx = canvas.getContext('2d');
const pianoKeysContainer = document.getElementById('pianoKeys');
const gridContainer = document.getElementById('gridContainer');
const trackSelect = document.getElementById('trackSelect');
const snapSelect = document.getElementById('snapSelect');

// Calculate total keys
const totalKeys = (CONFIG.endOctave - CONFIG.startOctave + 1) * 12;
const canvasHeight = totalKeys * CONFIG.noteHeight;
const canvasWidth = CONFIG.totalBeats * CONFIG.beatWidth;

// Initialize Canvas
canvas.width = canvasWidth;
canvas.height = canvasHeight;

// Tone.js Synths
const synths = {
    piano: new Tone.PolySynth(Tone.Synth).toDestination(),
    drums: new Tone.MembraneSynth().toDestination(), // Simple drum placeholder
    bass: new Tone.MonoSynth().toDestination(),
    strings: new Tone.PolySynth(Tone.AMSynth).toDestination()
};

// Helper: Pitch to Y coordinate
function getPitchY(pitch) {

    const startNote = CONFIG.startOctave * 12 + 12; // C2 = 36 (MIDI standard usually C4=60)
    
    const maxMidi = (CONFIG.endOctave + 1) * 12 + 11; // B6
    const minMidi = CONFIG.startOctave * 12; // C2
    

    
    const topMidi = (CONFIG.endOctave + 1) * 12 - 1; // B of endOctave
    const bottomMidi = CONFIG.startOctave * 12;      // C of startOctave
    
    return (topMidi - pitch) * CONFIG.noteHeight;
}

function getYPitch(y) {
    const topMidi = (CONFIG.endOctave + 1) * 12 - 1;
    const index = Math.floor(y / CONFIG.noteHeight);
    return topMidi - index;
}

// Time to X coordinate
function getTimeX(beat) {
    return beat * CONFIG.beatWidth;
}

function getXTime(x) {
    return x / CONFIG.beatWidth;
}

// Note names (chose sharps over flats for simplicity)
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function getNoteName(midi) {
    const note = NOTE_NAMES[midi % 12];
    const octave = Math.floor(midi / 12) - 1;
    return `${note}${octave}`;
}

// Initialize Piano Keys
function initPianoKeys() {
    const topMidi = (CONFIG.endOctave + 1) * 12 - 1;
    const bottomMidi = CONFIG.startOctave * 12;

    for (let i = topMidi; i >= bottomMidi; i--) {
        const key = document.createElement('div');
        const isBlack = [1, 3, 6, 8, 10].includes(i % 12);
        
        key.className = `piano-key ${isBlack ? 'black' : 'white'}`;
        key.style.height = `${CONFIG.noteHeight}px`;
        key.style.top = `${(topMidi - i) * CONFIG.noteHeight}px`;
        
        if (!isBlack) {
            key.innerText = getNoteName(i);
        }
        
        // Play note on click
        key.addEventListener('mousedown', () => {
            Tone.start();
            synths.piano.triggerAttackRelease(Tone.Frequency(i, "midi").toNote(), "8n");
        });

        pianoKeysContainer.appendChild(key);
    }
    // Adjust container height to match canvas
    // pianoKeysContainer.style.height = `${canvasHeight}px`; // REMOVED: Let CSS handle container height
}

// Drawing
function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw rows (pitches)
    for (let i = 0; i < totalKeys; i++) {
        const y = i * CONFIG.noteHeight;
        const midi = getYPitch(y);
        const isBlack = [1, 3, 6, 8, 10].includes(midi % 12);
        
        ctx.fillStyle = isBlack ? '#2a2a2a' : '#222';
        ctx.fillRect(0, y, canvas.width, CONFIG.noteHeight);
        
        ctx.strokeStyle = '#333';
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    // Draw columns (beats)
    for (let i = 0; i <= CONFIG.totalBeats; i++) {
        const x = i * CONFIG.beatWidth;
        ctx.strokeStyle = i % 4 === 0 ? '#555' : '#333';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
}

function drawNotes() {
    // Draw inactive tracks first (Onion Skinning)
    project.tracks.forEach(track => {
        if (track.id === project.activeTrackId) return;
        
        ctx.globalAlpha = 0.3; // Ghost effect
        ctx.fillStyle = track.color;
        ctx.strokeStyle = track.color;
        
        track.notes.forEach(note => {
            const x = getTimeX(note.startTime);
            const y = getPitchY(note.pitch);
            const w = note.duration * CONFIG.beatWidth;
            const h = CONFIG.noteHeight;
            
            ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
        });
        ctx.globalAlpha = 1.0;
    });

    // Draw active track
    const activeTrack = project.tracks[project.activeTrackId];
    
    activeTrack.notes.forEach(note => {
        const x = getTimeX(note.startTime);
        const y = getPitchY(note.pitch);
        const w = note.duration * CONFIG.beatWidth;
        const h = CONFIG.noteHeight;
        
        const isSelected = interactionState.selectedNotes.has(note);
        
        // Draw shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillRect(x + 3, y + 3, w - 2, h - 2);

        // Draw note body
        ctx.fillStyle = isSelected ? '#fff' : activeTrack.color;
        ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
        
        // Draw border
        ctx.strokeStyle = isSelected ? '#000' : '#fff';
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);

        // Draw resize handle visual
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillRect(x + w - 6, y + 1, 4, h - 2);
    });
    
    // Draw Selection Box
    if (interactionState.mode === 'select') {
        const box = interactionState.selectionBox;
        ctx.fillStyle = 'rgba(0, 168, 255, 0.2)';
        ctx.strokeStyle = '#00a8ff';
        ctx.lineWidth = 1;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeRect(box.x, box.y, box.w, box.h);
    }
}

function drawPlayhead() {
    if (!isPlaying) return;
    
    // Calculate x based on current transport position
    // This is a simplified visualization
    const x = (Tone.Transport.seconds / (60 / CONFIG.tempo)) * CONFIG.beatWidth;
    
    ctx.strokeStyle = '#FF5252';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.lineWidth = 1;
}

function render() {
    drawGrid();
    drawNotes();
    drawPlayhead();
    requestAnimationFrame(render);
}

// Interaction State
let interactionState = {
    mode: 'idle', // 'create', 'move', 'resize', 'select', 'idle'
    targetNote: null,
    startX: 0,
    startY: 0,
    initialNotes: new Map(), // Map<Note, {startTime, pitch, duration}>
    selectionBox: { x: 0, y: 0, w: 0, h: 0 },
    selectedNotes: new Set()
};

// Helper: Check if point is inside a note rect
function getNoteAt(x, y) {
    const beat = getXTime(x);
    const pitch = getYPitch(y);
    const activeNotes = project.tracks[project.activeTrackId].notes;
    
    // Find note that covers this time and pitch
    return activeNotes.find(n => 
        n.pitch === pitch && 
        n.startTime <= beat && 
        (n.startTime + n.duration) > beat
    );
}

// Helper: Check if point is near the right edge of a note (for resize)
function isResizeZone(x, y, note) {
    const noteX = getTimeX(note.startTime);
    const noteW = note.duration * CONFIG.beatWidth;
    const noteRight = noteX + noteW;
    
    // Check if within 10px of right edge
    return (x >= noteRight - 10 && x <= noteRight && 
            y >= getPitchY(note.pitch) && y < getPitchY(note.pitch) + CONFIG.noteHeight);
}

// Event Listeners
canvas.addEventListener('mousedown', handleMouseDown);
canvas.addEventListener('mousemove', handleMouseMove);
canvas.addEventListener('mouseup', handleMouseUp);
canvas.addEventListener('dblclick', handleDoubleClick);

// Sync Scroll
gridContainer.addEventListener('scroll', () => {
    pianoKeysContainer.scrollTop = gridContainer.scrollTop;
});

function handleMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const note = getNoteAt(x, y);
    const activeNotes = project.tracks[project.activeTrackId].notes;

    if (note) {
        // If clicking an unselected note without Shift, clear selection
        if (!interactionState.selectedNotes.has(note) && !e.shiftKey) {
            interactionState.selectedNotes.clear();
            interactionState.selectedNotes.add(note);
        } else if (e.shiftKey) {
            // Toggle selection
            if (interactionState.selectedNotes.has(note)) {
                interactionState.selectedNotes.delete(note);
            } else {
                interactionState.selectedNotes.add(note);
            }
        }

        if (isResizeZone(x, y, note)) {
            interactionState.mode = 'resize';
            canvas.style.cursor = 'ew-resize';
        } else {
            interactionState.mode = 'move';
            canvas.style.cursor = 'move';
        }

        interactionState.startX = x;
        interactionState.startY = y;
        
        // Store initial state for all selected notes
        interactionState.initialNotes.clear();
        interactionState.selectedNotes.forEach(n => {
            interactionState.initialNotes.set(n, { ...n });
        });

        // Play note
        Tone.start();
        const instrument = project.tracks[project.activeTrackId].instrument;
        synths[instrument].triggerAttackRelease(Tone.Frequency(note.pitch, "midi").toNote(), "8n");

    } else {
        // Clicked empty space
        if (!e.shiftKey) {
            interactionState.selectedNotes.clear();
        }

        // If Shift is held, start box selection
        if (e.shiftKey) {
            interactionState.mode = 'select';
            interactionState.startX = x;
            interactionState.startY = y;
            interactionState.selectionBox = { x, y, w: 0, h: 0 };
        } else {
            // Create new note
            interactionState.mode = 'create';
            const beat = Math.floor(getXTime(x)); // Snap to beat
            const pitch = getYPitch(y);
            
            const newNote = {
                pitch: pitch,
                startTime: beat,
                duration: 1
            };
            activeNotes.push(newNote);
            interactionState.selectedNotes.add(newNote);
            interactionState.targetNote = newNote;
            interactionState.startX = x;
            interactionState.startY = y;
            
            // Play note
            Tone.start();
            const instrument = project.tracks[project.activeTrackId].instrument;
            synths[instrument].triggerAttackRelease(Tone.Frequency(pitch, "midi").toNote(), "8n");
        }
    }
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (interactionState.mode === 'idle') {
        const note = getNoteAt(x, y);
        if (note) {
            canvas.style.cursor = isResizeZone(x, y, note) ? 'ew-resize' : 'move';
        } else {
            canvas.style.cursor = 'default';
        }
        return;
    }

    if (interactionState.mode === 'select') {
        const w = x - interactionState.startX;
        const h = y - interactionState.startY;
        interactionState.selectionBox = {
            x: w > 0 ? interactionState.startX : x,
            y: h > 0 ? interactionState.startY : y,
            w: Math.abs(w),
            h: Math.abs(h)
        };
        // TODO: Update selection based on intersection
        return;
    }

    const dx = x - interactionState.startX;
    const dy = y - interactionState.startY;
    const beatDelta = dx / CONFIG.beatWidth;
    const pitchDelta = -dy / CONFIG.noteHeight;

    if (interactionState.mode === 'resize') {
        interactionState.selectedNotes.forEach(note => {
            const initial = interactionState.initialNotes.get(note);
            if (!initial) return; // Should not happen for resize usually (single note)
            
            
            let newDuration = initial.duration + beatDelta;
            
            // Snap
            const snap = 1 / CONFIG.snap;
            newDuration = Math.round(newDuration / snap) * snap;
            if (newDuration <= 0) newDuration = snap;
            
            note.duration = newDuration;
        });
    } else if (interactionState.mode === 'create') {
        // Dragging to resize the newly created note
        const note = interactionState.targetNote;
        const startX = getTimeX(note.startTime);
        const newWidth = Math.max(CONFIG.beatWidth / 4, x - startX);
        let newDuration = newWidth / CONFIG.beatWidth;
        
        const snap = 1 / CONFIG.snap;
        newDuration = Math.round(newDuration / snap) * snap;
        if (newDuration <= 0) newDuration = snap;
        
        note.duration = newDuration;

    } else if (interactionState.mode === 'move') {
        interactionState.selectedNotes.forEach(note => {
            const initial = interactionState.initialNotes.get(note);
            
            let newStart = initial.startTime + beatDelta;
            let newPitch = initial.pitch + Math.round(pitchDelta);
            
            // Snap Start Time
            const snap = 1 / CONFIG.snap;
            newStart = Math.round(newStart / snap) * snap;
            if (newStart < 0) newStart = 0;
            
            // Clamp Pitch
            const maxMidi = (CONFIG.endOctave + 1) * 12 - 1;
            const minMidi = CONFIG.startOctave * 12;
            if (newPitch > maxMidi) newPitch = maxMidi;
            if (newPitch < minMidi) newPitch = minMidi;
            
            note.startTime = newStart;
            note.pitch = newPitch;
        });
    }
}

function handleMouseUp(e) {
    if (interactionState.mode !== 'idle') {
        saveState(); // Save history
    }
    
    interactionState.mode = 'idle';
    interactionState.targetNote = null;
    interactionState.initialNotes.clear();
    interactionState.selectionBox = { x: 0, y: 0, w: 0, h: 0 };
    canvas.style.cursor = 'default';
}

function handleDoubleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const note = getNoteAt(x, y);
    if (note) {
        const activeNotes = project.tracks[project.activeTrackId].notes;
        const index = activeNotes.indexOf(note);
        if (index > -1) {
            activeNotes.splice(index, 1);
            interactionState.selectedNotes.delete(note);
            saveState();
        }
    }
}

// Playback Controls
document.getElementById('playBtn').addEventListener('click', async () => {
    await Tone.start();
    
    if (isPlaying) return;
    
    isPlaying = true;
    Tone.Transport.bpm.value = CONFIG.tempo;
    
    // Schedule notes
    Tone.Transport.cancel(); // Clear previous
    
    project.tracks.forEach(track => {
        // Simple instrument mapping (could be expanded)
        // For now, everything uses the same synth but we could change parameters
        
        track.notes.forEach(note => {
            const time = note.startTime * (60 / CONFIG.tempo);
            const duration = note.duration * (60 / CONFIG.tempo);
            
            Tone.Transport.schedule((t) => {
                // Trigger sound
                synths[track.instrument].triggerAttackRelease(
                    Tone.Frequency(note.pitch, "midi").toNote(), 
                    duration, 
                    t
                );
            }, time);
        });
    });
    
    Tone.Transport.start();
});

document.getElementById('stopBtn').addEventListener('click', () => {
    isPlaying = false;
    Tone.Transport.stop();
});

document.getElementById('clearBtn').addEventListener('click', () => {
    // Clear active track only
    project.tracks[project.activeTrackId].notes = [];
    Tone.Transport.stop();
    isPlaying = false;
    saveState();
});

document.getElementById('generateBtn').addEventListener('click', async () => {
    const btn = document.getElementById('generateBtn');
    const originalText = btn.innerText;
    btn.innerText = "Generating...";
    btn.disabled = true;
    
    try {
        const activeTrack = project.tracks[project.activeTrackId];
        
        // Prepare notes (add default instrument)
        const notesToSend = activeTrack.notes.map(n => ({
            ...n,
            instrument: activeTrack.instrument
        }));
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: notesToSend })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log("Received notes:", data.notes);
            
            if (data.notes.length === 0) {
                alert("AI generated 0 notes. Try adding more context or wait for the model to train!");
            }

            // Update notes with validation
            const newNotes = [];
            let maxBeat = CONFIG.totalBeats;
            
            data.notes.forEach(n => {
                if (typeof n.pitch === 'number' && typeof n.startTime === 'number' && typeof n.duration === 'number') {
                    newNotes.push({
                        pitch: n.pitch,
                        startTime: n.startTime,
                        duration: n.duration
                    });
                    
                    // Expand canvas if needed
                    if (n.startTime + n.duration > maxBeat) {
                        maxBeat = n.startTime + n.duration;
                    }
                }
            });
            
            // Replace active track notes with generated result
            // The model returns the full sequence (context + generated)
            activeTrack.notes = newNotes;
            
            // Update canvas size if needed
            if (maxBeat > CONFIG.totalBeats) {
                CONFIG.totalBeats = Math.ceil(maxBeat + 4); // Add 1 bar padding
                canvas.width = CONFIG.totalBeats * CONFIG.beatWidth;
            }
            
            saveState();
            
        } else {
            alert("Error: " + data.message);
        }
        
    } catch (error) {
        console.error("Generation failed:", error);
        alert("Generation failed. Is the backend running?");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

document.getElementById('improveBtn').addEventListener('click', async () => {
    const btn = document.getElementById('improveBtn');
    
    if (interactionState.selectedNotes.size === 0) {
        alert("Please select some notes to improve first!");
        return;
    }
    
    const originalText = btn.innerText;
    btn.innerText = "Improving...";
    btn.disabled = true;
    
    try {
        const activeTrack = project.tracks[project.activeTrackId];
        
        // Calculate selection range
        let minStart = Infinity;
        let maxEnd = -Infinity;
        
        interactionState.selectedNotes.forEach(n => {
            if (n.startTime < minStart) minStart = n.startTime;
            if (n.startTime + n.duration > maxEnd) maxEnd = n.startTime + n.duration;
        });
        
        // Prepare notes
        const notesToSend = activeTrack.notes.map(n => ({
            ...n,
            instrument: activeTrack.instrument
        }));
        
        const response = await fetch('/api/improve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                notes: notesToSend,
                range: { start: minStart, end: maxEnd },
                temperature: 0.9, // Slightly lower temp for more coherent infilling
                steps: 50
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log("Received improved notes:", data.notes);
            
            // Update notes
            const newNotes = [];
            data.notes.forEach(n => {
                if (typeof n.pitch === 'number' && typeof n.startTime === 'number' && typeof n.duration === 'number') {
                    newNotes.push({
                        pitch: n.pitch,
                        startTime: n.startTime,
                        duration: n.duration
                    });
                }
            });
            
            activeTrack.notes = newNotes;
            interactionState.selectedNotes.clear(); // Clear selection as notes are new objects
            saveState();
            
        } else {
            alert("Error: " + data.message);
        }
        
    } catch (error) {
        console.error("Improvement failed:", error);
        alert("Improvement failed. Is the backend running?");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

// Initialize UI
function initUI() {
    // Populate Track Select
    trackSelect.innerHTML = '';
    project.tracks.forEach(track => {
        const option = document.createElement('option');
        option.value = track.id;
        option.innerText = track.name;
        trackSelect.appendChild(option);
    });
    
    trackSelect.value = project.activeTrackId;
    
    trackSelect.addEventListener('change', (e) => {
        project.activeTrackId = parseInt(e.target.value);
        interactionState.selectedNotes.clear();
        render();
    });
    
    snapSelect.addEventListener('change', (e) => {
        CONFIG.snap = parseInt(e.target.value);
    });
}

// Initialize
initUI();
initPianoKeys();
render();
