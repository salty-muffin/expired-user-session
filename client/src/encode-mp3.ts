import { Mp3Encoder } from '@breezystack/lamejs';

export const convertToMP3 = (audioBlob: Blob) => {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.readAsArrayBuffer(audioBlob);
		reader.onloadend = () => {
			try {
				const wavBuffer = reader.result;
				const wavData = new Uint8Array(wavBuffer as ArrayBuffer);

				// convert wav to mp3
				const mp3Encoder = new Mp3Encoder(1, 44100, 192); // 1 channel, 44.1 kHz, 128 kbps
				const samples = new Int16Array(wavData);
				const mp3Data = [];
				let remaining = samples.length;
				const sampleBlockSize = 1152;
				for (let i = 0; i < samples.length; i += sampleBlockSize) {
					const sampleChunk = samples.subarray(i, i + sampleBlockSize);
					const mp3Buf = mp3Encoder.encodeBuffer(sampleChunk);
					if (mp3Buf.length > 0) {
						mp3Data.push(new Int8Array(mp3Buf));
					}
					remaining -= sampleBlockSize;
				}
				const d = mp3Encoder.flush();
				if (d.length > 0) {
					mp3Data.push(new Int8Array(d));
				}

				const mp3Blob = new Blob(mp3Data, { type: 'audio/mp3' });
				resolve(mp3Blob);
			} catch (error) {
				reject(error);
			}
		};
	});
};
