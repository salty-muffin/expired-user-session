# ai-spirit-box

an ai version of the classic spirit box. can be used for post-internet transcommunication (i.e. communication with the spirits of dead users).

## setup

.env file needs to be created inside the server directory. inside the env file `PASSWORD=xxx` must be set to the password that the user should sign in with. the caddyfile must be adjusted to the proper domain name. currently docker compose needs to be run as root with `sudo docker compose up`.

## development requirements

- `ffmpeg` with `ffplay` needs to be installed.
- needs a conda environment (can be installed from environment.yml file).
- needs node to building the frontend (20.x.x+) ideally installed with `nvm`.
- for proper operation it needs a cuda comapitble gpu with necessary drivers installed.
- the backend should be operated behind a reverse proxy (e.g. `caddy`) or tunnel (e.g. `ngrok`).

## production requirements (docker)

- nvidia drivers need to be installed on the host machine.
- for cuda support [nvidia-conatiner-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) needs to be installed.

## to do

- [x] restructure (there should be a better structure than "modules/\*").
- [x] try implementation with [huggingface transformers](https://huggingface.co/docs/transformers/main/en/model_doc/bark) for [additional speedups](https://huggingface.co/blog/optimizing-bark).
- [x] split off ai generation into seperate process, with the server running in the main process. models will be loaded in the ai process. communication will happen through pipes and queues.
- [ ] try to split up model steps (semantic, course and fine) in huggingface for possibility of exiting the loop sooner. if not possible, revert to original bark?
- [ ] backup models, etc. (especially if using original bark).
- [ ] make dead-profiles page prettier.
- [x] pick right whisper model
- [ ] work over prompts and text generation method
