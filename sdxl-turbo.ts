import { findParentPreDOMNode } from 'lexical/nodes/LexicalTextNode'
import { ui_checkpoint, ui_sampler, run_ckpt_vae_clip, run_prompt } from './_prefab'

app({
    ui: (form) => ({
        prompt: form.prompt({
            label: 'Prompt',
            tooltip: 'SDXL Turbo only uses a positive prompt',
        }),
        checkpoint_name: ui_checkpoint(form),
        sampler_name: ui_sampler(form),
        steps: form.int({
            default: 1,
            label: 'Steps',
            min: 0,
            tooltip: 'SDXL Turbo only needs 1 or 2 steps',
            hideSlider: true,
        }),
        noise_seed: form.seed({
            label: 'Seed',
        }),
        cfg: form.int({
            default: 1,
            label: 'CFG',
            min: 0,
            tooltip: 'Should be 1',
            hideSlider: true,
        }),
    }),
    run: async (flow, p) => {
        const graph = flow.nodes
        let { model, vae, clip } = run_ckpt_vae_clip(flow, p.checkpoint_name)

        const { prompt, steps, noise_seed, cfg, sampler_name } = p

        const sigmas = graph.SDTurboScheduler({ model, steps })
        const sampler = graph.KSamplerSelect({ sampler_name })
        const latent_image = graph.EmptyLatentImage({
            width: 512,
            height: 512,
            batch_size: 1,
        })

        const positive = run_prompt(flow, { richPrompt: prompt, clip, ckpt: model, outputWildcardsPicked: true })
        const negative = graph.CLIPTextEncode({ clip: clip, text: '' })

        const ksampler = graph.SamplerCustom({
            model,
            positive: positive.conditionning,
            negative,
            sampler,
            sigmas,
            latent_image,
            noise_seed,
            cfg,
        })

        graph.SaveImage({
            filename_prefix: 'sdxl-turbo',
            images: graph.VAEDecode({ vae, samples: ksampler.outputs.output }),
        })

        await flow.PROMPT()
    },
})
