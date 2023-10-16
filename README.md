# Febrilucia
Acid shader that I put on Sakura in VRChat. I recommend trying it in nature maps with lots of greenery. Last update to this shader for that avatar was April 27th, 2021.

---
**If you use this shader in your world, please give me credit somewhere :D**, thanks

---

The source code here is a mess and it might look to some like it's obfuscated. However, I assure you, it's just from me making very half-assed edits to it every weekend over the course of a year (2020-2021).
Some of the uniforms are unused and were just being used to figure out what magic numbers I should hardcode into the shader. :D

Sorry about the mixed spacing, it looked fine in Notepad++ when I wrote it lol

I didn't use version control when I was writing the shader originally, but I do have two different copies of it stored in different projects from different points in time:
- one from September 11th, 2020
- one from April 27th, 2021

I've uploaded both here. Both are a mess, but the 2021 version is messier.

---
I played around with adding AudioLink support for a while, but got bored before I finished it. I've thrown these into the `AudioLink` folder here.

# Explanation of smeared blurring effect
I'd probably call this "chromatic smearing" or something, I don't know if there's already a name for it. Basically, you sample the texture (in this case the screen) at the regular UV, then reinterpret the sampled color as a vector and offset the UV using that vector. Rinse and repeat 15 times or so, blurring the resulting color each time. Because colors are mostly the same in both eyes, it just kind of works in VR.

I showed xwidghet this sometime later and then he made a cool fractal effect with the same sort of technique, but adding in worldspace depth to also distort by. Very trippy looking.

# The watery stuff running across surfaces
Similar trick, you can reinterpret the color sampled from the screen -> a UV to sample a normal map with. Then use the sampled normal to distort the screen UV before you do the smearing. Add a `+= _Time.y` to the UV's Y-coordinate before you sample the normal map and it'll give you a normal distortion that flows along a surface, because most surfaces are chromatically consistent looking. And colors that are close to each other sample the normal map in similar places, so as long as there's no big jump in the color, it'll distort similarly.

# Other notable things
- Colors shift in intensities
  - Just imagine a color wheel and rotating along the edge of it with time. Take the color it's on and bump up the value/brightness (as in HSV) of sampled colors that are close to that current color.
- Smooth interpolation between mirrored realities with some bit manipulation and funny graphs I drew in a notebook. One of the graphs looked like a mountain when graphed in 1D, see the `mountains` function.
- UVs used for screen sampling here get mirrored when they go outside the \[0,1] range, see the `mirrorSat` function.
- The number of steps done in chromatic smearing isn't a constant, it goes in and out like a wave. Sine is nice.
- Magic numbers are used in the timing of every effect being used so that they all go in and out independently without phases lining up. People noticing and understanding patterns is bad when you want interesting trippiness.
## In the 2021 file
- Visual snow was added in. In hindsight, I don't really like this effect very much, mainly because of video compression for wireless HMDs.
- Gravitational distortion around the center of the screen. In hindsight, I don't really like this very much either, I think it's kinda dumb.

---
If you can't tell, I liked the earlier versions of this shader more.
