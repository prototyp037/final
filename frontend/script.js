// Configuration
const CONFIG = {
    noteHeight: 20,
    beatWidth: 40,
    totalBeats: 32, // 8 bars of 4/4
    startOctave: 2,
    endOctave: 6,
    tempo: 120
};

// State
let notes = []; // Array of { pitch: number, startTime: number, duration: number }
let isPlaying = false;
let playbackId = null;
let currentBeat = 0;

// DOM Elements
const canvas = document.getElementById('pianoRollCanvas');
const ctx = canvas.getContext('2d');
const pianoKeysContainer = document.getElementById('pianoKeys');
const gridContainer = document.getElementById('gridContainer');

// Calculate total keys
const totalKeys = (CONFIG.endOctave - CONFIG.startOctave + 1) * 12;
const canvasHeight = totalKeys * CONFIG.noteHeight;
const canvasWidth = CONFIG.totalBeats * CONFIG.beatWidth;

// Initialize Canvas
canvas.width = canvasWidth;
canvas.height = canvasHeight;

// Tone.js Synth
const synth = new Tone.PolySynth(Tone.Synth).toDestination();

// Helper: Pitch to Y coordinate
function getPitchY(pitch) {
    // Pitch 0 is the highest note in our range (top of canvas)
    // We need to map MIDI pitch to Y.
    // Let's say startOctave C is at the bottom.
    // MIDI note for C2 is 36.
    const startNote = CONFIG.startOctave * 12 + 12; // C2 = 36 (MIDI standard usually C4=60)
    // Actually let's just use relative indices.
    // 0 is the lowest note (bottom), totalKeys-1 is highest (top).
    // But canvas Y=0 is top.
    // So Y = (totalKeys - 1 - (pitch - startNote)) * noteHeight
    
    // Let's simplify: index 0 is the highest note drawn at y=0
    // index totalKeys-1 is the lowest note drawn at bottom.
    // So if pitch index is 0 (highest), y=0.
    // We need to map MIDI pitch to this index.
    
    const maxMidi = (CONFIG.endOctave + 1) * 12 + 11; // B6
    const minMidi = CONFIG.startOctave * 12; // C2
    
    // Let's define the range explicitly
    // We want to draw from Top (High Pitch) to Bottom (Low Pitch)
    // High pitch: B6 (Midi 83) -> Y=0
    // Low pitch: C2 (Midi 24) -> Y=Height
    
    const topMidi = (CONFIG.endOctave + 1) * 12 - 1; // B of endOctave
    const bottomMidi = CONFIG.startOctave * 12;      // C of startOctave
    
    // y = (topMidi - pitch) * noteHeight
    return (topMidi - pitch) * CONFIG.noteHeight;
}

function getYPitch(y) {
    const topMidi = (CONFIG.endOctave + 1) * 12 - 1;
    const index = Math.floor(y / CONFIG.noteHeight);
    return topMidi - index;
}

// Helper: Time to X coordinate
function getTimeX(beat) {
    return beat * CONFIG.beatWidth;
}

function getXTime(x) {
    return x / CONFIG.beatWidth;
}

// Note names
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
            synth.triggerAttackRelease(Tone.Frequency(i, "midi").toNote(), "8n");
        });

        pianoKeysContainer.appendChild(key);
    }
    // Adjust container height to match canvas
    pianoKeysContainer.style.height = `${canvasHeight}px`;
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
    notes.forEach(note => {
        const x = getTimeX(note.startTime);
        const y = getPitchY(note.pitch);
        const w = note.duration * CONFIG.beatWidth;
        const h = CONFIG.noteHeight;
        
        // Draw shadow
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillRect(x + 3, y + 3, w - 2, h - 2);

        // Draw note body
        ctx.fillStyle = '#4CAF50';
        // Rounded rect simulation
        ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
        
        // Draw border
        ctx.strokeStyle = '#2E7D32';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);

        // Draw resize handle visual
        ctx.fillStyle = '#81C784';
        ctx.fillRect(x + w - 6, y + 1, 4, h - 2);
    });
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
    mode: 'idle', // 'create', 'move', 'resize', 'idle'
    targetNoteIndex: -1,
    startX: 0,
    startY: 0,
    initialNote: null, // Copy of note data at start of drag
    tempNote: null // For creation drag
};

// Helper: Check if point is inside a note rect
function getNoteAt(x, y) {
    const beat = getXTime(x);
    const pitch = getYPitch(y);
    
    // Find note that covers this time and pitch
    // Note: y check is simple because rows are discrete, but x needs range check
    // Actually, getYPitch returns the pitch row.
    
    return notes.findIndex(n => 
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

function handleMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left + gridContainer.scrollLeft;
    const y = e.clientY - rect.top + gridContainer.scrollTop;
    
    const noteIndex = getNoteAt(x, y);
    
    if (noteIndex !== -1) {
        const note = notes[noteIndex];
        if (isResizeZone(x, y, note)) {
            interactionState.mode = 'resize';
            canvas.style.cursor = 'ew-resize';
        } else {
            interactionState.mode = 'move';
            canvas.style.cursor = 'move';
        }
        interactionState.targetNoteIndex = noteIndex;
        interactionState.startX = x;
        interactionState.startY = y;
        interactionState.initialNote = { ...note };
        
        // Play note on select
        Tone.start();
        synth.triggerAttackRelease(Tone.Frequency(note.pitch, "midi").toNote(), "8n");
    } else {
        // Create new note
        interactionState.mode = 'create';
        const beat = Math.floor(getXTime(x));
        const pitch = getYPitch(y);
        
        const newNote = {
            pitch: pitch,
            startTime: beat,
            duration: 1
        };
        notes.push(newNote);
        interactionState.targetNoteIndex = notes.length - 1;
        interactionState.startX = x;
        interactionState.startY = y;
        interactionState.initialNote = { ...newNote };
        
        // Play note
        Tone.start();
        synth.triggerAttackRelease(Tone.Frequency(pitch, "midi").toNote(), "8n");
    }
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left + gridContainer.scrollLeft;
    const y = e.clientY - rect.top + gridContainer.scrollTop;
    
    // Cursor updates when not dragging
    if (interactionState.mode === 'idle') {
        const noteIndex = getNoteAt(x, y);
        if (noteIndex !== -1) {
            if (isResizeZone(x, y, notes[noteIndex])) {
                canvas.style.cursor = 'ew-resize';
            } else {
                canvas.style.cursor = 'move';
            }
        } else {
            canvas.style.cursor = 'default';
        }
        return;
    }
    
    const note = notes[interactionState.targetNoteIndex];
    if (!note) return;

    if (interactionState.mode === 'resize' || interactionState.mode === 'create') {
        // Calculate new duration
        const startX = getTimeX(note.startTime);
        const newWidth = Math.max(CONFIG.beatWidth / 4, x - startX); // Min width 1/4 beat
        const newDuration = newWidth / CONFIG.beatWidth;
        
        // Snap to 1/4 beat grid
        note.duration = Math.round(newDuration * 4) / 4;
        if (note.duration <= 0) note.duration = 0.25;
        
    } else if (interactionState.mode === 'move') {
        const dx = x - interactionState.startX;
        const dy = y - interactionState.startY;
        
        const beatDelta = dx / CONFIG.beatWidth;
        const pitchDelta = -dy / CONFIG.noteHeight; // Up is positive pitch, but negative Y
        
        // Calculate new start time (snap to 1/4 beat)
        let newStart = interactionState.initialNote.startTime + beatDelta;
        newStart = Math.round(newStart * 4) / 4;
        if (newStart < 0) newStart = 0;
        
        // Calculate new pitch
        let newPitch = interactionState.initialNote.pitch + Math.round(pitchDelta);
        // Clamp pitch
        const maxMidi = (CONFIG.endOctave + 1) * 12 - 1;
        const minMidi = CONFIG.startOctave * 12;
        if (newPitch > maxMidi) newPitch = maxMidi;
        if (newPitch < minMidi) newPitch = minMidi;
        
        note.startTime = newStart;
        if (note.pitch !== newPitch) {
            note.pitch = newPitch;
            // Optional: Play note when pitch changes
            // synth.triggerAttackRelease(Tone.Frequency(newPitch, "midi").toNote(), "16n");
        }
    }
}

function handleMouseUp(e) {
    interactionState.mode = 'idle';
    interactionState.targetNoteIndex = -1;
    interactionState.initialNote = null;
    canvas.style.cursor = 'default';
}

function handleDoubleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left + gridContainer.scrollLeft;
    const y = e.clientY - rect.top + gridContainer.scrollTop;
    
    const noteIndex = getNoteAt(x, y);
    if (noteIndex !== -1) {
        notes.splice(noteIndex, 1);
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
    notes.forEach(note => {
        const time = note.startTime * (60 / CONFIG.tempo);
        const duration = note.duration * (60 / CONFIG.tempo);
        Tone.Transport.schedule((t) => {
            synth.triggerAttackRelease(
                Tone.Frequency(note.pitch, "midi").toNote(), 
                duration, 
                t
            );
        }, time);
    });
    
    Tone.Transport.start();
});

document.getElementById('stopBtn').addEventListener('click', () => {
    isPlaying = false;
    Tone.Transport.stop();
});

document.getElementById('clearBtn').addEventListener('click', () => {
    notes = [];
    Tone.Transport.stop();
    isPlaying = false;
});

document.getElementById('generateBtn').addEventListener('click', () => {
    alert("AI Generation feature coming soon! This will connect to the Python backend.");
});

// Initialize
initPianoKeys();
render();
