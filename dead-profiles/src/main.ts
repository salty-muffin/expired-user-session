import './style.css';
import { paths } from './profiles.json';

const SCROLLINTERVAL = 20;
const MAXSCROLL = 500;
const DELAY = 5000; // 5 seconds delay

let currentIndex = 0;
let scrollFinished = true;

const frame = document.getElementById('frame') as HTMLIFrameElement | null;

const loadNextFile = (files: string[]) => {
	if (frame) {
		if (scrollFinished) {
			scrollFinished = false;

			frame.src = `/profiles/${files[currentIndex]}`;
			currentIndex++;
			if (currentIndex >= paths.length) currentIndex = 0;
			frame.onload = () => {
				setTimeout(scrollThrough, DELAY);
			};
		}
		setTimeout(() => loadNextFile(files), 100); // check again in 100 milliseconds, if the page has finished scrolling
	}
};

const scrollThrough = () => {
	if (frame?.contentWindow) {
		console.log(frame.contentWindow?.document.body.scrollHeight);
		if (
			frame.contentWindow!.scrollY + document.body.offsetHeight >=
				frame.contentWindow!.document.body.scrollHeight ||
			frame.contentWindow!.scrollY >= MAXSCROLL
		) {
			scrollFinished = true;
		} else {
			frame.contentWindow.scrollBy(0, 1);
			setTimeout(scrollThrough, SCROLLINTERVAL);
		}

		frame.contentWindow.document.onscroll = () => {
			console.log(
				frame.contentWindow!.scrollY,
				frame.contentWindow!.scrollY + document.body.offsetHeight
			);
		};
	}
};

if (paths.length > 0) {
	loadNextFile(paths);
} else {
	console.log('No HTML files found.');
}
