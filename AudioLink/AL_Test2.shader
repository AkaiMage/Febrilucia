Shader "RedMage/Test2" {
    Properties {
		_Water ("Water", 2D) = "bump" {}
		_AudioBumpSample ("bumpy", range(0,1)) = .05
		_AudioBumpSmoothLow ("low", range(0,1)) = 0
		_AudioBumpSmoothHigh ("high", range(0,1)) = 1
	}
    SubShader {
        Tags { "Queue"="Transparent+3" }
		
		GrabPass { "_Jenkem" }

        Pass {
			Cull Off
			ZTest Always
			
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
			#pragma target 5.0

            #include "UnityCG.cginc"
			#include "AudioLink.cginc"

            struct appdata {
                float4 vertex : POSITION;
            };

            struct v2f {
                float4 pos : SV_POSITION;
				float4 projPos : TEXCOORD0;
				float3 posWorld : TEXCOORD1;
            };

			//SamplerState sampler_AudioGraph_Point_Clamp;
			//Texture2D<float4> _AudioTexture;
			
			sampler2D _Jenkem;
			float4 _Jenkem_TexelSize;
			
			float _AudioBumpSmoothLow, _AudioBumpSmoothHigh;
			float _AudioBumpSample;
			sampler2D _Water;
			
			float3 hsv2rgb(float3 c) {
				return ((saturate(abs(frac(c.x+float3(0,2./3,1./3))*6-3)-1)-1)*c.y+1)*c.z;
			}
			
			float3 rgb2hsv(float3 c) {
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

				float d = q.x - min(q.w, q.y);
				float e = 1.0e-10;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}
			
			float mountain(float t) {
				return 1 - abs(1 - frac(t * 2));
			}
			
			float2 mirrorFlipper(float2 uv) {
				return 1 - abs(1 - 2 * frac(uv * .5));
			}
			
			float fakeIntegral(float a, float b, float dx) {
				return (min(a, b) + abs(a - b) * .5) * dx;
			}

            v2f vert(appdata v) {
                v2f o;
				
				// disable mirror rendering
				if (unity_CameraProjection[2][0] != 0 || unity_CameraProjection[2][1] != 0) v.vertex = 1e25;
				
				float4 world = mul(unity_ObjectToWorld, v.vertex);
				o.posWorld = world.xyz;
                o.pos = UnityWorldToClipPos(world);
				o.projPos = ComputeGrabScreenPos(o.pos);
                return o;
            }
			
			float audioSample(bool has_audio, float w, uint2 delayBand) {
				float2 uv = (delayBand + float2(.5, .5)) / float2(4, 32);
				if (has_audio) return AudioLinkData(uv.xy);
				return .5 + .5 * sin((_Time.y + w) * 1.28384572728 * uv.x + uv.y);
			}

            float2x2 rotmat(float a) {
                float s = sin(a), c = cos(a);
                return float2x2(c, -s, s, c);
            }
			
			float2 computeSphereUV(float3 worldSpacePos) {
				float3 viewDir = normalize(worldSpacePos - _WorldSpaceCameraPos);
				float lat = acos(viewDir.y);
				float lon = atan2(viewDir.z, viewDir.x);
				lon = fmod(lon + UNITY_PI, UNITY_TWO_PI) - UNITY_PI;
				return 1 - float2(lon, lat) / UNITY_PI;
			}
			
			float spectroSample(float2 uv) {
				//uv.y += .1 * uv.x + _Time.y;
				//uv.y *= uv.x;
				uv = frac(uv);
				float a = 2 * (2 * AudioLinkLerp(ALPASS_DFT + float2(uv.x * AUDIOLINK_ETOTALBINS, 0)).z - .5);
				float b = 2 * (2 * AudioLinkLerp(ALPASS_DFT + float2(uv.y * AUDIOLINK_ETOTALBINS, 0)).z - .5);
				return smoothstep(_AudioBumpSmoothLow, _AudioBumpSmoothHigh, a + b);
			}
			
			float3 calculateAcid(bool has_audio, float2 uv) {
				if (has_audio) {
					float2 uvo = uv;
					uv *= 10;
					float off = _AudioBumpSample;
					float left = spectroSample(uv - float2(off, 0));
					float right = spectroSample(uv + float2(off, 0));
					float up = spectroSample(uv - float2(0, off));
					float down = spectroSample(uv + float2(0, off));
					float dx = right - left;
					float dy = down - up;
					float dz = sqrt(1 - dx * dx - dy * dy);
					float2 asdf = normalize(float2(dx, dy));
					dx = asdf.x;
					dy = asdf.y;
					//float dx = ddx(combined);
					//float dy = ddy(combined);
					
					//float3 r = cross(float3(dx, dy, combined));
					return lerp(UnpackNormal(smoothstep(0.2,1,tex2Dlod(_Water, float4(uvo, 0, 0)))),
					            smoothstep(.2, 1, -2 * float3(dx, dy, dz)),
								.5);
					//return AudioLinkLerpMultiline(ALPASS_WAVEFORM + float2(uv.x * 200, 0)).r;
				}
				//return .1*float3(sin(uv.y), 0, 3.1*cos(uv.x));
				return UnpackNormal(smoothstep(0.2,1,tex2Dlod(_Water, float4(uv, 0, 0))));
			}

            float4 frag(v2f i) : SV_Target {
				half2 testwh;
				_AudioTexture.GetDimensions(testwh.x, testwh.y);
				
				bool has_audio = (testwh.x > 16);
				
				// audio texture is available
				float band1Latest = audioSample(has_audio, 1, ALPASS_AUDIOLINK + uint2(0, 0));
				float band2Latest = audioSample(has_audio, 2, ALPASS_AUDIOLINK + uint2(0, 1));
				
				float2 grabUV = i.projPos.xy / i.projPos.w;
				float2 origGrabUV = grabUV;
				
				float4 grabCol = tex2D(_Jenkem, grabUV);
				
				float2 sphereUV = computeSphereUV(i.posWorld);
                sphereUV.y += _Time.y * .14567;
				//sphereUV += 2 * (2 * grabCol.rg - 1);
				
				float r = sin(.25*_Time.y)*smoothstep(0,1,abs(cos(.25*_Time.y)));

				float3 boop_val = calculateAcid(has_audio, sphereUV);
				//return float4(.5 * boop_val + .5, 1);
                float2 distort = mul(rotmat(boop_val.z), boop_val.xy) * .025 * (has_audio ? 1 : saturate(sin(_Time.y * .1234589 + .234678)));
				grabUV += distort;
				grabUV.x += r*sin(_Time.y * 1.0991283484 + band1Latest + 40.12934845) * .001;
				grabUV.y += cos(_Time.y * 3.123 + .1 + band2Latest) * .001;
				grabCol = tex2D(_Jenkem, grabUV);
				
				float4 grabbbb = grabCol;
				
				float theredchannelacc = grabCol.r;
				float thegreenchannelacc = grabCol.g * .1;
				
				float audio_differential = has_audio ?
					saturate(audioSample(has_audio, 3, ALPASS_AUDIOLINK + uint2(0, 1)) - audioSample(has_audio, 4, ALPASS_AUDIOLINK + uint2(3, 1))) :
					0;
				float2x2 rotty = rotmat(audio_differential * .1);
				
				float mover = saturate((audioSample(has_audio, 5, ALPASS_AUDIOLINK + uint2(0, 3)) + audioSample(has_audio, 6, ALPASS_AUDIOLINK + uint2(0, 2))) * audioSample(has_audio, 7, ALPASS_AUDIOLINK + uint2(0, 2)));
				
				#define COUNTY 15
				
				UNITY_UNROLL for (uint j = 0; j < COUNTY; ++j) {
					float2 moveVec = (.05 + .05 * audio_differential) * (grabCol.rg * 2 - 1);
					moveVec = mul(rotty, moveVec);
					moveVec *= .62 * mover * .1 / COUNTY * (.75 + .25 * cos(_Time.y) + .5);
                    moveVec += grabCol.b * .001 * float2(sin(_Time.y), cos(_Time.y));
					grabUV += moveVec;
					
					grabCol = tex2D(_Jenkem, mirrorFlipper(grabUV));
					grabbbb += grabCol;
					theredchannelacc += grabCol.r;
					thegreenchannelacc += grabCol.g * (.05 + .05 * grabCol.b);
				}
				
				grabbbb /= COUNTY+1;
				
				grabCol.rgb = rgb2hsv(grabCol.rgb);
				
				float band4_average = has_audio ?
                    smoothstep(0, 1, lerp(audioSample(has_audio, 8, ALPASS_AUDIOLINK + uint2(0, 3)), audioSample(has_audio, 9, ALPASS_AUDIOLINK + uint2(16, 3)), .25)) :
					.1 * abs(sin(_Time.y * .199991 + 92.192948));
				
				grabCol.x += thegreenchannelacc * .75 + band4_average * .1;
				float colorIntenser = 0;
                {
                    float a = frac(_Time.y * .3374 + .489539);
                    float c = grabCol.x;
                    if (a > c) colorIntenser = min(a - c, 1 + c - a);
                    else colorIntenser = min(c - a, 1 + a - c);
                    colorIntenser = 3 * smoothstep(0, 0.5, colorIntenser);
                }
				//grabCol.z += grabCol.z * .2 * audioSample(has_audio, 10, uint2(0, uint(32 * grabUV.x - origGrabUV.x)));
				grabCol.x = frac(grabCol.x);
				grabCol.z += colorIntenser * saturate(grabCol.z);
				grabCol.rgb = hsv2rgb(grabCol.rgb);
				
				return grabCol;
            }
            ENDCG
        }
    }
}
