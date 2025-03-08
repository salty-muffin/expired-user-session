import './style.css';

const baseUrl = import.meta.env.BASE_URL;

import config from './config.yml';
import profiles from '../../data/profiles.json';
import moveData from './mouse-simulation/movedata.json';
import scrollData from './mouse-simulation/scrolldata.json';

let currentIndex = 0;

const MAX_DISPLAY_TIME_ACTIVE =
	Boolean(config.maxDisplayTime) || Boolean(config.maxDisplayTime?.max);

const frame = document.getElementById('frame') as HTMLIFrameElement | null;
const cursor = document.getElementById('cursor') as HTMLIFrameElement | null;
const browserUi = document.getElementById('browser-ui') as HTMLIFrameElement | null;
const urlProtocol = document.getElementById('protocol') as HTMLIFrameElement | null;
const urlHostname = document.getElementById('hostname') as HTMLIFrameElement | null;
const urlPath = document.getElementById('path') as HTMLIFrameElement | null;

interface Range {
	min: number;
	max: number;
}

const getRandomInt = (attribute: { min: number; max: number }) => {
	const min = Math.ceil(attribute.min);
	const max = Math.floor(attribute.max);
	return Math.floor(Math.random() * (max - min + 1)) + min;
};

const checkProbability = (chance: number) => {
	chance = Math.min(1 - Number.EPSILON, Math.max(0, chance));
	return Math.random() < chance;
};

interface MouseData {
	x: number;
	y: number;
	t: number;
}

const shuffleArray = (array: any[]) => {
	return array
		.map((value) => ({ value, sort: Math.random() }))
		.sort((a, b) => a.sort - b.sort)
		.map(({ value }) => value);
};

const splitMouseData = (
	data: MouseData[],
	range: Range,
	shuffle = false,
	invertSegmentChance = 0,
	noInverseFirstSegment = false
) => {
	let result: MouseData[][] = [];
	let currentSubarray: MouseData[] = [];
	let currentTimeRange = getRandomInt(range);
	let baseTime = 0;
	let invert = noInverseFirstSegment
		? false
		: invertSegmentChance && checkProbability(invertSegmentChance);

	for (const element of data) {
		// Reset the subarray if time exceeds the current time range
		if (element.t - baseTime >= currentTimeRange) {
			result.push(currentSubarray);

			// Prepare for the next subarray
			result = result.filter((dataArray) => dataArray.length);
			baseTime += result.at(-1)?.at(-1)?.t ?? 0;
			currentSubarray = [];
			currentTimeRange = getRandomInt(range);

			// Check whether to invert or not
			invert = Boolean(invertSegmentChance) && checkProbability(invertSegmentChance);
		}
		currentSubarray.push({
			x: !invert ? element.x : -element.x,
			y: !invert ? element.y : -element.y,
			t: element.t - baseTime
		});
	}

	// Handle the last subarray
	if (currentSubarray.length && currentSubarray.at(-1)) {
		result.push(currentSubarray);
	}

	result = result.filter((dataArray) => dataArray.length);
	if (result.length < 2 || !shuffle) {
		return result;
	}
	if (noInverseFirstSegment) {
		return [result[0], ...shuffleArray(result.slice(1))];
	}
	return shuffleArray(result);
};

let scrollSegments: MouseData[][] = [];
let currentScrollSegment: MouseData[] = [];
let scrollStart: number;
let scrollPosition = { x: 0, y: 0 };
const scroll = (timestamp: number) => {
	// Remix data if data is empty (first time or used up)
	if (!scrollSegments.length) {
		scrollSegments = splitMouseData(
			scrollData,
			config.scrollSegmentTime,
			config.shuffleScroll,
			config.invertSegmentChance,
			config.noInverseFirstSegment
		);
	}
	// Reload next segment if current segment is used up
	if (!currentScrollSegment.length) {
		currentScrollSegment = scrollSegments.shift() ?? [];
		scrollStart = timestamp;

		// & possibly jump a bit on the page
		if (config.jumpChance && checkProbability(config.jumpChance) && frame?.contentWindow) {
			scrollPosition.y += getRandomInt(config.jumpPerSegment);

			frame.contentWindow.scrollTo({
				left: scrollPosition.x,
				top: scrollPosition.y,
				behavior: 'instant'
			});
		}
	}

	// Get the last scroll delta for the current timestamp
	let currentData: MouseData | null = null;
	while (currentScrollSegment.length && currentScrollSegment[0].t <= timestamp - scrollStart) {
		currentData = currentScrollSegment.shift() ?? null;
	}
	// Get absolute scroll position and scroll to it
	if (frame?.contentWindow && currentData) {
		scrollPosition = {
			x: Math.min(Math.max((currentData.x as number) + scrollPosition.x, 0), bodyDimensions.x),
			y: Math.min(Math.max((currentData.y as number) + scrollPosition.y, 0), bodyDimensions.y)
		};
		frame.contentWindow.scrollTo({
			left: scrollPosition.x,
			top: scrollPosition.y,
			behavior: 'smooth'
		});
	}
	requestAnimationFrame(scroll);
};

// Check if an element is clickable
function isClickable(element: Element | null): boolean {
	while (element) {
		const clickableTags = ['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'LABEL'];
		if (clickableTags.includes(element.tagName)) {
			return true;
		}

		// Check for explicit `onclick` or `role="button"`
		const el = element as HTMLElement;
		if (typeof el.onclick === 'function' || el.getAttribute('role') === 'button') {
			return true;
		}

		// Check if element has `pointer` cursor style
		const computedStyle = window.getComputedStyle(element);
		if (computedStyle.cursor === 'pointer') {
			return true;
		}

		// Move to the parent element
		element = element.parentElement;
	}
	return false;
}

let moveSegments: MouseData[][] = [];
let currentMoveSegment: MouseData[] = [];
let moveStart: number;
const moveMouse = (timestamp: number) => {
	// Remix data if data is empty (first time or used up)
	if (!moveSegments.length) {
		moveSegments = splitMouseData(moveData, config.mouseMoveSegmentTime, config.shuffleMouse);
	}
	if (!currentMoveSegment.length) {
		currentMoveSegment = moveSegments.shift() ?? [];
		moveStart = timestamp;
	}

	// Get the last mouse position for the current timestamp
	let currentData: MouseData | null = null;
	while (currentMoveSegment.length && currentMoveSegment[0].t <= timestamp - moveStart) {
		currentData = currentMoveSegment.shift() ?? null;
	}
	// Get mouse position adjusted to iframe dimensions and move the mouse there
	if (frame && browserUi && cursor && currentData) {
		const cursorX = (currentData.x as number) * frame.offsetWidth;
		const cursorY = (currentData.y as number) * frame.offsetHeight + browserUi.offsetHeight;
		cursor.style.left = `${cursorX}px`;
		cursor.style.top = `${cursorY}px`;
		if (frame.contentDocument) {
			const elementUnderCursor = frame.contentDocument.elementFromPoint(cursorX, cursorY);
			if (isClickable(elementUnderCursor)) {
				cursor.classList.remove('default');
				cursor.classList.add('pointer');
			} else {
				cursor.classList.remove('pointer');
				cursor.classList.add('default');
			}
		}
	}
	requestAnimationFrame(moveMouse);
};

let bodyDimensions = { x: 0, y: 0 };
let first = true;
let timeoutId: NodeJS.Timeout | null;
const loadNextFile = (profiles: { path: string; url: string; character: string }[]) => {
	// Reset the display time if it is active
	if (timeoutId) clearTimeout(timeoutId);

	if (frame) {
		scrollPosition = { x: 0, y: 0 };

		if (urlProtocol && urlHostname && urlPath) {
			try {
				const url = new URL(profiles[currentIndex].url);

				urlProtocol.innerText = url.protocol + '//';
				urlHostname.innerText = url.hostname;
				urlPath.innerText = url.port + url.pathname;
			} catch {
				console.warn(`invalid url: ${profiles[currentIndex].url}`);
			}
		}

		frame.src = `/profiles/${baseUrl.replace('/', '')}/${profiles[currentIndex].path}`;
		frame.onload = () => {
			if (frame?.contentDocument) {
				bodyDimensions = {
					x: frame.contentDocument.body.offsetWidth,
					y: frame.contentDocument.body.offsetHeight
				};
			}
			if (first) {
				// start animations if it's the first time the frame is loaded
				requestAnimationFrame(scroll);
				if (config.enableMouse) requestAnimationFrame(moveMouse);

				first = false;
			}
		};
	}

	// If maxDisplayTime is set, load next page after a random time
	if (MAX_DISPLAY_TIME_ACTIVE) {
		timeoutId = setTimeout(() => {
			currentIndex++;
			if (currentIndex >= profiles.length) currentIndex = 0;

			loadNextFile(profiles);

			// Send the POST request with the index and filepath
			fetch('/control', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ index: currentIndex, path: profiles[currentIndex].path })
			}).catch((error: Error) => {
				console.error('Error sending control command:', error);
			});
		}, getRandomInt(config.maxDisplayTime));
	}
};

// Add keyboard event listeners for arrow key controls to send a signal to the server to cycle through the profiles
document.addEventListener('keydown', (event: KeyboardEvent) => {
	// Prevent default behavior (like scrolling with arrow keys)
	if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
		event.preventDefault();

		// Set the index and cycle through the profiles
		if (event.key === 'ArrowLeft') {
			currentIndex--;
			if (currentIndex < 0) currentIndex = profiles.length - 1;
		} else if (event.key === 'ArrowRight') {
			currentIndex++;
			if (currentIndex >= profiles.length) currentIndex = 0;
		}
		loadNextFile(profiles);

		console.log(currentIndex); // DEBUG
		// Send the POST request with the index and filepath
		fetch('/control', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ index: currentIndex, path: profiles[currentIndex].path })
		}).catch((error: Error) => {
			console.error('Error sending control command:', error);
		});
	}
});

if (profiles.length > 0) {
	loadNextFile(profiles);
} else {
	console.log('No HTML files found.');
}
