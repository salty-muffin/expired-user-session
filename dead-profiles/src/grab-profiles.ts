import fs from 'fs';
import path from 'path';

// Define the directory where your HTML files are located
const pagesDir = path.join(__dirname, '..', 'public', 'profiles');

// Define the output JSON file path
const outputFile = path.join(__dirname, 'profiles.json');

// Function to generate the JSON file with the list of HTML files
function generateFilesJson() {
	// Read the contents of the pages directory
	fs.readdir(pagesDir, (err, files) => {
		if (err) {
			console.error('Error reading the directory:', err);
			return;
		}

		// Filter the files to include only .html files
		const htmlFiles = files.filter((file) => file.endsWith('.html'));

		// Prepare the JSON object
		const jsonData = {
			paths: htmlFiles
		};

		// Write the JSON data to the output file
		fs.writeFile(outputFile, JSON.stringify(jsonData, null, 2), (err) => {
			if (err) {
				console.error('Error writing the JSON file:', err);
			} else {
				console.log(`files.json has been created with ${htmlFiles.length} HTML files.`);
			}
		});
	});
}

// Run the function to generate files.json
generateFilesJson();
