let lines: String[] = [];
let cursor: HTMLDivElement | null = null;

export const log = (element: HTMLElement | null = null, ...args: any) => {
	if (args.length && typeof args[0] == 'string' && args[0].length) console.log(...args);
	lines.push(args.map((arg: any) => String(arg)).join(' '));
	if (element) element.innerHTML = lines.join('<br />');
	if (cursor) removeCursor();
	// scroll to bottom
	window.scrollTo(0, document.body.scrollHeight);
};

export const logCursor = (element: HTMLElement | null = null, ...args: any) => {
	console.log(...args);
	lines.push(args.map((arg: any) => String(arg)).join(' '));
	if (element) {
		element.innerHTML = lines.join('<br />');
		addCursor(element);
	}
	// scroll to bottom
	window.scrollTo(0, document.body.scrollHeight);
};

export const clearLog = (element: HTMLElement | null = null) => {
	lines = [];
	if (element) element.innerHTML = '';
	if (cursor) removeCursor();
	// scroll to top
	window.scrollTo(0, 0);
};

const addCursor = (element: HTMLElement) => {
	const dimensions = getCharacterDimensions();
	cursor = document.createElement('div');
	cursor.setAttribute('id', 'cursor');
	cursor.style.width = `${dimensions.width}px`;
	cursor.style.height = `${dimensions.height}px`;
	cursor.style.top = `${(lines.length - 1) * dimensions.height}px`;
	cursor.style.left = `${lines[lines.length - 1].length * dimensions.width}px`;
	element.parentElement?.appendChild(cursor);
};

const removeCursor = () => {
	cursor?.remove();
	cursor = null;
};

const getCharacterDimensions = (parent = document.body) => {
	const span = document.createElement('span');
	span.style.visibility = 'hidden';
	span.style.position = 'fixed';
	span.textContent = '0';

	parent.appendChild(span);
	const rect = span.getBoundingClientRect();
	const dimensions = { width: rect.width, height: rect.height };
	parent.removeChild(span);

	return dimensions;
};
