import './style.css';
import { paths } from './profiles.json';

const scrollInterval = { min: 5, max: 30 };
let currentScrollInterval = 0;
const scrollBy = { min: -2, max: 2 };
let currentScrollBy = 0;
const maxScrollTime = { min: 8000, max: 20000 };
let currentMaxScrollTime = 0;
let currentStartTime = 0;
const delay = { min: 10, max: 50 }; // 5 seconds delay

const fadeDelay = 2 * 60 * 1000;
const fadeTime = 8 * 60;

const resetChance = 0.001;

let currentIndex = 0;
let scrollFinished = true;

const frame = document.getElementById('frame') as HTMLIFrameElement | null;
const cover = document.getElementById('cover') as HTMLDivElement | null;

const urlParams = new URLSearchParams(window.location.search);
console.log('fading:', Boolean(urlParams.get('fade')));

if (urlParams.get('fade') && cover) {
	setTimeout(() => {
		console.log('Fading now.');
		cover.style.transition = `opacity ${fadeTime}s linear`;
		cover.classList.add('fade');
	}, fadeDelay);
}

const getRandomInt = (attribute: { min: number; max: number }) => {
	const min = Math.ceil(attribute.min);
	const max = Math.floor(attribute.max);
	return Math.floor(Math.random() * (max - min + 1)) + min;
};

const resetValues = () => {
	// set random values
	currentScrollInterval = getRandomInt(scrollInterval);
	currentScrollBy = getRandomInt(scrollBy);

	console.log(
		'currentScrollInterval',
		currentScrollInterval,
		'currentScrollBy',
		currentScrollBy,
		'currentMaxScrollTime',
		currentMaxScrollTime
	);
};

const loadNextFile = (files: string[]) => {
	if (frame) {
		if (scrollFinished) {
			scrollFinished = false;

			frame.src = `/profiles/${files[currentIndex]}`;
			currentIndex++;
			if (currentIndex >= paths.length) currentIndex = 0;
			frame.onload = () => {
				// start at random point in the page
				if (frame.contentWindow)
					frame.contentWindow.scrollBy(
						0,
						getRandomInt({ min: 0, max: frame.contentWindow.document.body.scrollHeight })
					);

				resetValues();
				currentMaxScrollTime = getRandomInt(maxScrollTime);
				currentStartTime = new Date().getTime();
				setTimeout(scrollThrough, getRandomInt(delay));
			};
		}
		setTimeout(() => loadNextFile(files), 100); // check again in 100 milliseconds, if the page has finished scrolling
	}
};

const scrollThrough = () => {
	if (frame?.contentWindow) {
		if (
			frame.contentWindow!.scrollY + document.body.offsetHeight >=
				frame.contentWindow!.document.body.scrollHeight ||
			frame.contentWindow!.scrollY <= 0 ||
			new Date().getTime() >= currentStartTime + currentMaxScrollTime
		) {
			scrollFinished = true;
		} else {
			frame.contentWindow.scrollBy(0, currentScrollBy);
			if (Math.random() <= resetChance) resetValues();
			setTimeout(scrollThrough, currentScrollInterval);
		}
	}
};

if (paths.length > 0) {
	loadNextFile(paths);
} else {
	console.log('No HTML files found.');
}
