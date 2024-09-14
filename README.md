# ai-spirit-box

an ai version of the classic spirit box

## requirements

- needs a conda environment (can be installed from environment.yml file).
- needs node to building the frontend (20.x.x+) ideally installed with `nvm`.
- for proper operation it needs a cuda comapitble gpu with necessary drivers installed.
- the backend should be operated behind a reverse proxy (e.g. `caddy`) or tunnel (e.g. `ngrok`).

## to do

- restructure (there should be a better structure than "modules/*").
- try implementation with [huggingface transformers](https://huggingface.co/docs/transformers/main/en/model_doc/bark) for [additional speedups](https://huggingface.co/blog/optimizing-bark).
- try to split up model steps (semantic, course and fine) in huggingface for possibility of exiting the loop sooner. if not possible, revert to original bark?
- split off ai generation into seperate process, with the server running in the main process. models will be loaded in the ai process. communication will happen through pipes and queues.
- backup models, etc. (especially if using original bark).
- make dead-profiles page prettier.
