import { defineConfig } from 'vite';
import viteYaml from '@modyfi/vite-plugin-yaml';

export default defineConfig({
	base: './',
	esbuild: {
		supported: {
			'top-level-await': true //browsers can handle top-level-await features
		}
	},
	plugins: [viteYaml()]
});
