Shader "RedMage/Hallucidity/Febrilucia" {
	Properties {
		[Enum(UnityEngine.Rendering.CullMode)] _CullMode("Cull Mode", Float) = 0
		[Enum(UnityEngine.Rendering.CompareFunction)] _ZTest ("ZTest", Int) = 4
		
        [Enum(A,0,B,1,C,2,D,3,Off,4)] _FixedPhase ("Fixed Phase", Int) = 4
        _DarkControl ("Darkness", Range(0, 1)) = 0
		_SlideyBoi ("Slide me 1", Range(-1, 1)) = 0
        _SlideyBoi2 ("Slide me 2", Range(-1, 1)) = 0
        _MainTex ("Main tesx", 2D) = "white" {}
        _BoopMap ("funny water texture thing go here", 2D) = "bump" {}
        _BoopQuantity ("how much water go boop", Float) = 0
        _MainTexBurnIn ("Main Tex Burn In", Range(0, 1)) = 0
	}
	SubShader {
		Tags { "RenderType"="Transparent" "Queue" = "Overlay+2" }
		
		Cull [_CullMode]
		ZTest [_ZTest]
		
		GrabPass { "_Garbasd" }

		Pass {
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata {
				float4 vertex : POSITION;
			};

			struct v2f {
				float4 pos : SV_POSITION;
				float3 posWorld : TEXCOORD0;
				float4 projPos : TEXCOORD1;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			
			sampler2D _Garbasd;
			float4 _Garbasd_TexelSize;

            float _SlideyBoi, _SlideyBoi2;

            sampler2D _BoopMap;
            float4 _BoopMap_ST;

            uint _FixedPhase;

            float _BoopQuantity;

            float _DarkControl;
            float _MainTexBurnIn;
			
			float3 hsv2rgb(float3 c) {
				return ((clamp(abs(frac(c.x+float3(0,.666,.333))*6-3)-1,0,1)-1)*c.y+1)*c.z;
			}
			
			float3 rgb2hsv(float3 c) {
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

				float d = q.x - min(q.w, q.y);
				float e = 1.0e-10;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}

            float2 mirrorSat(float2 uv) {
                if (uv.x < 0) uv.x = -uv.x;
                else if (uv.x > 1) uv.x = -1 + uv.x;
                
                if (uv.y < 0) uv.y = -uv.y;
                else if (uv.y > 1) uv.y = -1 + uv.y;
                return uv;
            }
			
			v2f vert (appdata v) {
				v2f o;
                float3 head = _WorldSpaceCameraPos.xyz;
				#if defined(USING_STEREO_MATRICES)
                head = lerp(unity_StereoWorldSpaceCameraPos[0], unity_StereoWorldSpaceCameraPos[1], .5);
                #endif

                if (unity_CameraProjection[2][0] != 0 || unity_CameraProjection[2][1] != 0) {
                    v.vertex = 1e25;
                } else {
                    v.vertex.xyz *= 8;
                }

                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
				o.pos = UnityObjectToClipPos(v.vertex);
				o.projPos = ComputeGrabScreenPos(o.pos);
				return o;
			}

            float mountains(float t) {
                float asdf = frac(t);
                if (asdf > .5) asdf = 1 - asdf;
                asdf *= 2;
                return asdf;
            }

            float3 colorBurn(float3 b, float3 s) {
                float3 res;
                UNITY_UNROLL for (int i = 0; i < 3; ++i) {
                    if (b[i] == 1) res[i] = 1;
                    else if (s[i] == 0) res[i] = 0;
                    else res[i] = 1 - min(1, (1 - b[i]) / s[i]);
                }
                return res;
            }

            float2x2 rotmat(float a) {
                float s = sin(a), c = cos(a);
                return float2x2(c, -s, s, c);
            }

            float3 baba(float3 co, float redthing, float time, float3 sstex) {
                co = rgb2hsv(co);

                float colorIntenser = 0;
                {
                    float a = frac(time * .6374 + .489539);
                    float c = co.x;
                    if (a > c) colorIntenser = min(a - c, 1 + c - a);
                    else colorIntenser = min(c - a, 1 + a - c);
                    colorIntenser = smoothstep(0, 0.5, colorIntenser);
                }

                co.x += redthing;
                co.x += .1 * sin(time / 2);
                co.x = frac(co.x);
                co.z = max(co.z * .5 * (1 - _DarkControl), pow(co.z, 1 + 1 * (.5 + .5*cos(time + .65))));
                co.z += colorIntenser * saturate(co.z);
                co = hsv2rgb(co);

                co = lerp(co, colorBurn(co, sstex), _MainTexBurnIn * (.5 + .5 * sin(time * .12345666666666 + .34241234)));
                return co;
            }

            float3 calculatePass(float4 grabCol, float2 grabUV, float2 ssuv, float time, uint flip, uint cnt, out float3 prevColor, float finalStepInterpolant) {
                float redAcc = 0;
                float redPrev = 0;

                float3 col = 0;

                ssuv += 2 * (2 * grabCol.rg - 1);
                ssuv.y += frac(time * .214567);


                float3 boop_val = UnpackNormal(tex2Dlod(_BoopMap, float4(ssuv, 0, 0)));
                float2 distort = mul(rotmat(boop_val.z), boop_val.xy) * _BoopQuantity * saturate(sin(time * .1234589 + .234678));

                grabUV += distort;

                for (uint q = 0; q < cnt; ++q) {
                    redPrev = redAcc;
                    redAcc += grabCol.r;
                    grabCol.rgb = hsv2rgb(float3(rgb2hsv(grabCol.rgb).x, 1, 1));
                    float2 displace = normalize(grabCol.rg * 2 - 1);
                    displace *= .62 * .05 / cnt * (.75 + .25 * cos(time));
                    displace += grabCol.b * .0005 * float2(sin(time), cos(time));
                    if (q == cnt - 1) displace *= finalStepInterpolant;

                    grabUV += displace;
                    grabUV = mirrorSat(grabUV);
                    grabUV = frac(grabUV);

                    float3 asdf = grabCol.rgb;
                    float abra = (flip & 1) != 0 ? -.5 : .5;
                    float kadabra = (flip & 2) != 0 ? -.5 : .5;

                    if (asdf.r * asdf.b < abra) {
                        grabUV.x = 1 - grabUV.x;
                    }
                    if (asdf.g < kadabra) {
                        grabUV.y = 1 - grabUV.y;
                    }
                    
                    grabCol = tex2D(_Garbasd, grabUV);
                    prevColor = col;
                    col += grabCol.rgb;
                }

                prevColor /= cnt - 1;
                redPrev /= cnt - 1;
                
                grabCol.rgb = col / cnt;
                redAcc /= cnt;

                float3 sstex = tex2Dlod(_MainTex, float4(ssuv + distort, 0, 0));
                //prevColor *= sstex;
                //grabCol.rgb *= sstex;
                prevColor = baba(prevColor, redPrev, time, sstex);
                return baba(grabCol.rgb, redAcc, time, sstex);
            }
			
			fixed4 frag (v2f i) : SV_Target {
				float3 viewSpace = mul(UNITY_MATRIX_V, i.posWorld.xyz - _WorldSpaceCameraPos);
                viewSpace.xy /= viewSpace.z;
				float2 grabUV = i.projPos.xy / i.projPos.w;
				float4 grabCol = tex2D(_Garbasd, grabUV);

                float width = _Garbasd_TexelSize.x;
                #if defined(USING_STEREO_MATRICES)
                width *= .5;
                #endif
                float height = _Garbasd_TexelSize.y;
                
                float2 ssuv = .5 * (1 - viewSpace.xy * float2((height*(width+1))/(width*(height+1)), 1));

                float3 colll = 0;

                const uint cnt = 13;

                float timeParam =  _Time.y / 2;

                uint phaseCount = 4;

                float3 prevColor;

                float t = fmod(_Time.y * .3333333, phaseCount * 2);
                uint ft = (uint) (floor(t) + .5);

                if (_FixedPhase != 4) ft = ((uint) _FixedPhase) << 1;
                if ((ft & 1) == 0) {


                    // now we do the subphase transition
                    float tt = frac(t);
                    float subp = fmod(tt * cnt * 2, cnt * 2);
                    uint pp = (uint) (floor(subp) + .5);
                    float fac = frac(subp);
                    uint actualCnt = 0;
                    bool up = false;
                    if (pp < cnt) {
                        actualCnt = cnt + pp;
                    } else {
                        actualCnt = cnt + (2 * cnt - pp);
                        fac = 1 - fac;
                    }
                    uint phase = ft >> 1;
                    fac = smoothstep(0, 1, fac);
                    colll = calculatePass(grabCol, grabUV, ssuv, timeParam, phase, actualCnt, prevColor, fac);
                    
                    colll = lerp(prevColor, colll, fac);
                } else {
                    uint phase_a = ft >> 1;
                    uint phase_b = phase_a + 1;
                    float factor = frac(t);
                    float3 res_a = calculatePass(grabCol, grabUV, ssuv, timeParam, phase_a, cnt, prevColor, 1);
                    float3 res_b = calculatePass(grabCol, grabUV, ssuv, timeParam, phase_b, cnt, prevColor, 1);
                    colll = lerp(res_a, res_b, factor);
                }

                return float4(colll, 1);
			}
			ENDCG
		}
	}
}