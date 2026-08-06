using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class BuildReplay {
    public static void BuildMac() {
        PlayerSettings.colorSpace = ColorSpace.Gamma;
        PlayerSettings.runInBackground = true;
        string root = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
        string output = Path.Combine(root, "unity_build/AlphaRacerReplay.app");
        Directory.CreateDirectory(Path.GetDirectoryName(output));
        Directory.CreateDirectory("Assets/Resources");
        Directory.CreateDirectory("Assets/Resources/ReplayAssets");
        ConfigureTexture("Assets/ThirdParty/PolyHaven/AsphaltTrack/asphalt_track_diff_1k.jpg", false);
        ConfigureTexture("Assets/ThirdParty/PolyHaven/AsphaltTrack/asphalt_track_nor_gl_1k.jpg", true);
        ConfigureTexture("Assets/ThirdParty/PolyHaven/LeafyGrass/leafy_grass_diff_1k.jpg", false);
        ConfigureTexture("Assets/ThirdParty/PolyHaven/LeafyGrass/leafy_grass_nor_gl_1k.jpg", true);
        CreateLitMaterial(
            "Assets/Resources/ReplayRoad.mat",
            "Assets/ThirdParty/PolyHaven/AsphaltTrack/asphalt_track_diff_1k.jpg",
            "Assets/ThirdParty/PolyHaven/AsphaltTrack/asphalt_track_nor_gl_1k.jpg",
            new Color(0.66f, 0.68f, 0.72f), 0.16f, 0.85f
        );
        CreateLitMaterial(
            "Assets/Resources/ReplayGround.mat",
            "Assets/ThirdParty/PolyHaven/LeafyGrass/leafy_grass_diff_1k.jpg",
            "Assets/ThirdParty/PolyHaven/LeafyGrass/leafy_grass_nor_gl_1k.jpg",
            new Color(0.57f, 0.66f, 0.54f), 0.08f, 0.65f
        );
        CreateLitMaterial("Assets/Resources/ReplayLitTemplate.mat", null, null, Color.white, 0.28f, 0f);
        CreateLitMaterial("Assets/Resources/ReplayCurbRed.mat", null, null, new Color(0.82f, 0.035f, 0.025f), 0.3f, 0f);
        CreateLitMaterial("Assets/Resources/ReplayCurbWhite.mat", null, null, new Color(0.9f, 0.9f, 0.86f), 0.25f, 0f);
        CreateTireMarkMaterial();
        CreateSkyMaterial();
        CopyReplayAsset("Assets/ThirdParty/KenneyCarKit/Models/FBX format/race.fbx", "Assets/Resources/ReplayAssets/RaceCar.fbx");
        CopyReplayAsset("Assets/ThirdParty/KenneyCarKit/Models/FBX format/race-future.fbx", "Assets/Resources/ReplayAssets/FutureCar.fbx");
        const string materialPath = "Assets/Resources/ReplayMaterial.mat";
        Shader shader = Shader.Find("Unlit/Color");
        if (shader == null) throw new System.Exception("Unlit/Color shader not found");
        Material replayMaterial = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
        if (replayMaterial == null) {
            AssetDatabase.CreateAsset(new Material(shader), materialPath);
            AssetDatabase.SaveAssets();
        } else {
            replayMaterial.shader = shader;
            EditorUtility.SetDirty(replayMaterial);
            AssetDatabase.SaveAssets();
        }
        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        const string scenePath = "Assets/Replay.unity";
        EditorSceneManager.SaveScene(scene, scenePath);
        BuildPlayerOptions options = new BuildPlayerOptions {
            scenes = new [] { scenePath },
            locationPathName = output,
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None
        };
        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded) throw new System.Exception("Unity build failed: " + report.summary.result);
        Debug.Log("Built replay app at " + output);
    }

    static void CopyReplayAsset(string source, string destination) {
        if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(source) == null) {
            throw new System.Exception("Required replay asset is missing: " + source);
        }
        if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(destination) != null) {
            AssetDatabase.DeleteAsset(destination);
        }
        if (!AssetDatabase.CopyAsset(source, destination)) {
            throw new System.Exception("Could not copy replay asset: " + source);
        }
    }

    static void ConfigureTexture(string path, bool normalMap) {
        TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
        if (importer == null) throw new System.Exception("Texture is missing: " + path);
        TextureImporterType desired = normalMap ? TextureImporterType.NormalMap : TextureImporterType.Default;
        importer.wrapMode = TextureWrapMode.Repeat;
        importer.filterMode = FilterMode.Trilinear;
        importer.anisoLevel = 8;
        if (importer.textureType != desired) {
            importer.textureType = desired;
            importer.SaveAndReimport();
        } else {
            importer.SaveAndReimport();
        }
    }

    static void CreateLitMaterial(string path, string albedoPath, string normalPath, Color tint, float smoothness, float bumpScale) {
        Shader shader = Shader.Find("Standard");
        if (shader == null) throw new System.Exception("Standard shader not found");
        Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null) {
            material = new Material(shader);
            AssetDatabase.CreateAsset(material, path);
        }
        material.shader = shader;
        material.color = tint;
        material.SetFloat("_Glossiness", smoothness);
        material.SetFloat("_Metallic", 0f);
        material.mainTexture = string.IsNullOrEmpty(albedoPath) ? null : AssetDatabase.LoadAssetAtPath<Texture2D>(albedoPath);
        Texture2D normal = string.IsNullOrEmpty(normalPath) ? null : AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath);
        material.SetTexture("_BumpMap", normal);
        material.SetFloat("_BumpScale", bumpScale);
        if (normal != null) material.EnableKeyword("_NORMALMAP"); else material.DisableKeyword("_NORMALMAP");
        EditorUtility.SetDirty(material);
        AssetDatabase.SaveAssets();
    }

    static void CreateSkyMaterial() {
        const string path = "Assets/Resources/ReplaySky.mat";
        Shader shader = Shader.Find("Skybox/Procedural");
        if (shader == null) throw new System.Exception("Procedural skybox shader not found");
        Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null) {
            material = new Material(shader);
            AssetDatabase.CreateAsset(material, path);
        }
        material.shader = shader;
        material.SetColor("_SkyTint", new Color(0.42f, 0.62f, 0.82f));
        material.SetColor("_GroundColor", new Color(0.32f, 0.34f, 0.36f));
        material.SetFloat("_AtmosphereThickness", 0.82f);
        material.SetFloat("_Exposure", 1.05f);
        EditorUtility.SetDirty(material);
        AssetDatabase.SaveAssets();
    }

    static void CreateTireMarkMaterial() {
        const string texturePath = "Assets/Resources/ReplayTireTread.asset";
        Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
        if (texture == null) {
            texture = new Texture2D(64, 64, TextureFormat.RGBA32, false);
            AssetDatabase.CreateAsset(texture, texturePath);
        }
        Color[] pixels = new Color[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                bool centerGroove = x >= 28 && x <= 35;
                int diagonal = x < 32 ? y + x : y + (63 - x);
                bool treadBlock = diagonal % 18 < 11;
                bool shoulderGap = (x < 7 || x > 56) && y % 16 < 4;
                float alpha = !centerGroove && treadBlock && !shoulderGap ? 0.68f : 0f;
                pixels[y * 64 + x] = new Color(0.015f, 0.017f, 0.02f, alpha);
            }
        }
        texture.SetPixels(pixels);
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.Apply(false, false);
        EditorUtility.SetDirty(texture);

        const string materialPath = "Assets/Resources/ReplayTireMark.mat";
        Shader shader = Shader.Find("Unlit/Transparent");
        if (shader == null) throw new System.Exception("Unlit/Transparent shader not found");
        Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
        if (material == null) {
            material = new Material(shader);
            AssetDatabase.CreateAsset(material, materialPath);
        }
        material.shader = shader;
        material.mainTexture = texture;
        material.color = Color.white;
        EditorUtility.SetDirty(material);
        AssetDatabase.SaveAssets();
    }
}
