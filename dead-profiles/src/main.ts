import './style.css';

import config from './config.yml';
import { paths } from './profiles.json';
import moveData from './mouse-simulation/movedata.json';
import scrollData from './mouse-simulation/scrolldata.json';

let currentIndex = 0;

const frame = document.getElementById('frame') as HTMLIFrameElement | null;
const cursor = document.getElementById('cursor') as HTMLIFrameElement | null;

interface Range {
	min: number;
	max: number;
}

const getRandomInt = (attribute: { min: number; max: number }) => {
	const min = Math.ceil(attribute.min);
	const max = Math.floor(attribute.max);
	return Math.floor(Math.random() * (max - min + 1)) + min;
};

interface MouseData {
	x: number;
	y: number;
	t: number;
}

function splitMouseData(data: MouseData[], range: Range): MouseData[][] {
	let result: MouseData[][] = [];
	let currentSubarray: MouseData[] = [];
	let currentTimeRange = getRandomInt(range);
	let baseTime = 0;

	for (const element of data) {
		// Reset the subarray if time exceeds the current time range
		if (element.t - baseTime >= currentTimeRange) {
			result.push(currentSubarray);

			// Prepare for the next subarray
			result = result.filter((dataArray) => dataArray.length);
			baseTime += result.at(-1)?.at(-1)?.t ?? 0;
			currentSubarray = [];
			currentTimeRange = getRandomInt(range);
		}
		currentSubarray.push({ ...element, t: element.t - baseTime });
	}

	// Handle the last subarray
	if (currentSubarray.length && currentSubarray.at(-1)) {
		result.push(currentSubarray);
	}

	return result.filter((dataArray) => dataArray.length);
}

let scrollSegments: MouseData[][] = [];
let currentScrollSegment: MouseData[] = [];
let scrollStart: number;
let scrollPosition = { x: 0, y: 0 };
const scroll = (timestamp: number) => {
	// Remix data if data is empty (first time or used up)
	if (!scrollSegments.length) {
		scrollSegments = splitMouseData(scrollData, config.scrollSegmentTime);
	}
	if (!currentScrollSegment.length) {
		currentScrollSegment = scrollSegments.shift() ?? [];
		scrollStart = timestamp;
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
		moveSegments = splitMouseData(moveData, config.mouseMoveSegmentTime);
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
	// Get mouse position adjusted to window dimensions and move the mouse there
	if (cursor && currentData) {
		const cursorX = (currentData.x as number) * window.innerWidth;
		const cursorY = (currentData.y as number) * window.innerHeight;
		cursor.style.left = `${cursorX}px`;
		cursor.style.top = `${cursorY}px`;
		if (frame?.contentDocument) {
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
const loadNextFile = (files: string[]) => {
	if (frame) {
		frame.src = `/profiles/${files[currentIndex]}`;
		currentIndex++;
		if (currentIndex >= paths.length) currentIndex = 0;
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

	// Load next page after a random time
	// setTimeout(() => loadNextFile(files), getRandomInt(config.maxDisplayTime));
};

if (paths.length > 0) {
	loadNextFile(paths);
} else {
	console.log('No HTML files found.');
}
