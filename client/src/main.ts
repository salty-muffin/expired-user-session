import '@fontsource/ubuntu-mono';
import './style.css';

import { log, logCursor } from './logging';

import { io } from 'socket.io-client';

// settings
const INTERVAL = 0;
const PLAYALL = false;

// global variables
let isRecording = false;
let isFirstResponse = false;
let audioChunks: BlobPart[] = [];
let mediaRecorder: MediaRecorder;
let audioBlobQueue: Blob[] = [];

let interval: number | null = null;
let timeout: number | null = null;

// elements
const player = document.getElementById('player') as HTMLAudioElement | null;
const terminal = document.getElementById('terminal') as HTMLPreElement | null;
const passwordInput = document.getElementById('password') as HTMLInputElement | null;
if (passwordInput) passwordInput.value = '';

// request permission for microphone and video
let stream: MediaStream;
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
	try {
		stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
	} catch (error) {
		console.error(`The following getUserMedia error occurred: '${error}'.`);
	}
else log(terminal, 'getUserMedia not supported on your browser!');

log(terminal, terminal?.innerText ?? '');
log(terminal, '');

logCursor(terminal, 'Password: ');
// keep password element in focus
const focusPassword = () => {
	passwordInput?.focus();
};
focusPassword();
document.addEventListener('click', focusPassword);
// capture password and start the main script if it has been catured
const getPassword = async (event: KeyboardEvent) => {
	if (event.key === 'Enter' && passwordInput) {
		passwordInput.removeEventListener('keydown', getPassword);
		document.removeEventListener('click', focusPassword);
		// passwordInput.removeEventListener('blur', focusPassword);
		console.log(passwordInput.value);
		// run the rest of your script
		await main(passwordInput.value);
	}
};
passwordInput?.addEventListener('keydown', getPassword);

const main = async (pass: string) => {
	// websocket connection with socket.io
	const socket = io(window.location.host, {
		auth: { user: 'seance', password: pass }
	});

	// connect to the server and get the video seed when ready
	socket.on('connect', async () => {
		log(terminal, 'Connected to server, obtaining video seed...');
		await getSeedFromCamera();
		setTimeout(() => {
			log(terminal, "Press 'space' to start recording a question, release to stop.");
		}, 1000);
	});

	socket.on('disconnect', () => {
		log(terminal, 'Lost connection to the server.');
	});

	socket.on('connect_error', (error) => {
		if (!socket.active) {
			// the connection was denied by the server
			log(terminal, error.message);
		}
	});

	// handle responses
	socket.on('first_response', (data) => {
		handleServerResponse(data);
		isFirstResponse = false;
	});

	socket.on('response', (data) => {
		if (!isFirstResponse || PLAYALL) handleServerResponse(data);
	});

	// event listeners for spacebar key press/release
	window.addEventListener('keydown', (event) => {
		if (event.code === 'Space' && !isRecording) {
			isRecording = true;
			startRecording();
		}
	});

	window.addEventListener('keyup', (event) => {
		if (event.code === 'Space' && isRecording) {
			isRecording = false;
			stopRecording();

			// start playback loop
			interval = setInterval(playQueue, 100);
		}
	});

	const playQueue = () => {
		if (player?.paused && audioBlobQueue.length) {
			timeout = setTimeout(() => {
				const audioBlob = audioBlobQueue.shift();
				if (audioBlob) {
					const audioUrl = URL.createObjectURL(audioBlob);
					player.src = audioUrl;
					player.play();
				}
			}, INTERVAL);
		}
	};

	// start recording audio
	const startRecording = () => {
		if (stream) {
			// stop playback
			if (player) {
				player.pause();
				player.currentTime = 0;
			}
			if (interval) {
				clearInterval(interval);
				interval = null;
			}
			if (timeout) {
				clearTimeout(timeout);
				timeout = null;
			}
			// clear queue
			audioBlobQueue = [];

			// record new audio
			mediaRecorder = new MediaRecorder(stream);
			audioChunks = [];

			mediaRecorder.ondataavailable = (event) => {
				audioChunks.push(event.data);
			};

			mediaRecorder.onstop = () => {
				log(terminal, 'Recording stopped.');
				const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
				audioChunks = [];
				sendAudioToServer(audioBlob);
				isFirstResponse = true;
			};

			mediaRecorder.start();
			log(terminal, 'Recording started...');
		} else throw Error('No MediaStream found for recording.');
	};

	// stop recording audio
	const stopRecording = () => {
		if (mediaRecorder && mediaRecorder.state === 'recording') {
			mediaRecorder.stop();
		}
	};

	// send the recorded audio to the server
	const sendAudioToServer = (audioBlob: Blob) => {
		// const mp3Blob = convertToMP3(audioBlob);
		const reader = new FileReader();
		reader.readAsArrayBuffer(audioBlob);
		reader.onloadend = () => {
			const audioBuffer = reader.result;
			socket.emit('contact', audioBuffer);
		};
	};

	// handle audio response from server and play it back
	const handleServerResponse = (data: BinaryData) => {
		console.log('Received response.');
		audioBlobQueue.push(new Blob([data], { type: 'audio/mp3' }));
	};

	// video capture and seed calculation
	const getSeedFromCamera = async () => {
		if (stream) {
			const video = document.createElement('video');
			video.srcObject = stream;
			video.play();

			video.addEventListener('canplay', () => {
				const canvas = document.createElement('canvas');
				const context = canvas.getContext('2d');
				canvas.width = video.videoWidth;
				canvas.height = video.videoHeight;
				if (context) {
					context.drawImage(video, 0, 0, canvas.width, canvas.height);
					const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
					const pixels = imageData.data;

					let seed = 0;
					for (let i = 0; i < pixels.length; i += 4) {
						seed += pixels[i]; // sum the red channel for simplicity
					}

					log(terminal, `Generated seed: ${seed}.`);
					socket.emit('seed', { seed: seed });
				}
				// stop the video stream
				stream.getTracks().forEach((track) => {
					if (track.kind === 'video') {
						track.stop();
					}
				});
				video.srcObject = null;
				video.remove();
			});
		} else throw Error('No MediaStream found for capturing.');
	};
};
