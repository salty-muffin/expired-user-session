import './style.css';

import config from './config.yml';
import { paths } from './profiles.json';
import moveData from './mouse-simulation/movedata.json';
import scrollData from './mouse-simulation/scrolldata.json';

let currentIndex = 0;

const frame = document.getElementById('frame') as HTMLIFrameElement | null;

const getRandomInt = (attribute: { min: number; max: number }) => {
	const min = Math.ceil(attribute.min);
	const max = Math.floor(attribute.max);
	return Math.floor(Math.random() * (max - min + 1)) + min;
};

const loadNextFile = (files: string[]) => {
	if (frame) {
		frame.src = `/profiles/${files[currentIndex]}`;
		currentIndex++;
		if (currentIndex >= paths.length) currentIndex = 0;
		frame.onload = () => {};
	}

	// Load next page after a random time
	setTimeout(() => loadNextFile(files), getRandomInt(config.maxDisplayTime));
};

interface MouseData {
	x: number;
	y: number;
	t: number;
}

const scroll = () => {
	// Remix Data
	const scrollSegments: MouseData[][] = [];
	const scrollDataCopy = [...scrollData];
	let lastElementT = 0;
	let currentElement = scrollDataCopy.shift();

	while (scrollDataCopy.length) {
		let currentScrollSegmentTime = getRandomInt(config.scrollSegmentTime);
		scrollSegments.push([]);
		while (currentElement && currentElement.t - lastElementT < currentScrollSegmentTime) {
			scrollSegments[scrollSegments.length - 1].push({
				...currentElement,
				t: currentElement.t - lastElementT
			});
			if (scrollDataCopy.length) break;
			currentElement = scrollDataCopy.shift();
		}
		lastElementT = currentElement?.t ?? 0;
	}

	// Restart
	scroll();
};

const moveMouse = () => {
	// Restart
	moveMouse();
};

if (paths.length > 0) {
	loadNextFile(paths);

	scroll();
	moveMouse();
} else {
	console.log('No HTML files found.');
}
