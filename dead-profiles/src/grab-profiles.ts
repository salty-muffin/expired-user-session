import fs from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';

// Define the directory where your HTML files are located
const pagesDir = path.join(__dirname, '..', 'public', 'profiles');

// Define the output JSON file path
const outputFile = path.join(__dirname, '..', '..', 'data', 'profiles.json');

// Function to extract title and first comment (holds url info if the page is downloaded with SingleFile) from an HTML file
const extractInfoFromHtml = async (filePath: string) => {
	try {
		const content = await fs.promises.readFile(filePath, 'utf8');

		// Create a DOM from the HTML content
		const dom = new JSDOM(content);

		// Extract the title
		const title = dom.window.document.querySelector('title')?.textContent || '';

		// Extract the first comment - JSDOM exposes comment nodes
		let comment = '';
		const nodeIterator = dom.window.document.createNodeIterator(
			dom.window.document,
			dom.window.NodeFilter.SHOW_COMMENT
		);

		// Get the first comment if it exists
		const firstComment = nodeIterator.nextNode();
		if (firstComment) {
			comment = firstComment.nodeValue?.trim() || '';
		}

		return { firstComment: comment, title };
	} catch (error) {
		console.error(`Error processing file ${filePath}:`, error);
		return { firstComment: '', title: '' };
	}
};

// FUnction to parse a SingleFile URL form the first comment
const parseSingleFileURL = (comment: string) => {
	const matches = comment.match(/(https?:\/\/[^ ]*)/);
	if (matches && matches.length > 1) {
		return matches[1];
	}
	return '';
};

// Function to generate the JSON file with the list of HTML files
const generateFilesJson = async () => {
	try {
		// Read the contents of the pages directory
		const files = await fs.promises.readdir(pagesDir);

		// Filter the files to include only .html files
		const htmlFiles = files.filter((file) => file.endsWith('.html'));

		// Process each HTML file to extract info
		const jsonData = await Promise.all(
			htmlFiles.map(async (fileName) => {
				const filePath = path.join(pagesDir, fileName);
				const { firstComment, title } = await extractInfoFromHtml(filePath);

				return {
					path: fileName,
					url: parseSingleFileURL(firstComment),
					name: '',
					character: '',
					title
				};
			})
		);

		// Write the JSON data to the output file
		await fs.promises.writeFile(outputFile, JSON.stringify(jsonData, null, 2));
		console.log(`profiles.json has been created with ${htmlFiles.length} HTML files.`);
	} catch (err) {
		console.error('Error generating files JSON:', err);
	}
};

// Run the function to generate files.json
await generateFilesJson();
