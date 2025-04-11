import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Философия картографии',
			description: 'Философия картографии: университетский курс. Картография и наука до 20 века. Современная картографическая наука. Теория картографии и теоретические концепции в картографии.',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: true,
			},
			customCss: [
				// Fontsource files for to regular and semi-bold font weights.
				'@fontsource/playfair-display/400.css',
				// Relative path to your custom CSS file
				'./src/styles/custom.css',
			],
			defaultLocale: 'root',
			locales: {
				root: {
					label: 'Русский',
					lang: 'ru',
				}
			},
			head: [
				{
					tag: 'script',
					attrs: {
						src: "//gc.zgo.at/count.js",
						'data-goatcounter': "https://poc.goatcounter.com/count",
						async: true
					}
				}
			],
			social: {
				email: 'mailto:titovgs@my.msu.ru',
				github: 'https://github.com/gtitov/poc'
			},
			sidebar: [
				{
					label: 'Курс',
					autogenerate: { directory: 'course' },
				},
				{
					label: 'Эссе',
					autogenerate: { directory: 'essays' },
				},
				// {
				// 	label: 'Рефераты',
				// 	autogenerate: { directory: 'papers' },
				// },
				{
					label: 'Рецензии',
					autogenerate: { directory: 'reviews' },
				},
				{
					label: 'Внешние ресурсы',
					collapsed: true,
					items: [
						{
							label: "Кафедра картографии и геоинформатики ↗",
							link: 'https://carto.geogr.msu.ru',
							attrs: { target: '_blank' },
						},
						{
							label: "Географический факультет ↗",
							link: 'https://geogr.msu.ru',
							attrs: { target: '_blank' },
						},
						{
							label: "МГУ имени М. В. Ломоносова ↗",
							link: 'https://msu.ru',
							attrs: { target: '_blank' },
						},
					],
				},
			],
			editLink: {
				baseUrl: 'https://github.com/gtitov/poc/edit/main/',
			},
		}),
	],
});
