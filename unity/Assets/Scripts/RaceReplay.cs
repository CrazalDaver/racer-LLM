using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

[Serializable] public class ReplayFrame { public float[] poses; public float[] throttle; }
[Serializable] public class ReplayData {
    public float fps = 30f;
    public float trackWidth = 1f;
    public float vehicleLength = 0.36f;
    public float vehicleWidth = 0.24f;
    public float[] trackX;
    public float[] trackY;
    public float[] racelineX;
    public float[] racelineY;
    public ReplayFrame[] frames;
}

class TireTrail {
    public Mesh mesh;
    public readonly List<Vector3> vertices = new List<Vector3>();
    public readonly List<Vector2> uv = new List<Vector2>();
    public readonly List<int> triangles = new List<int>();
    public readonly Vector3[] previous = new Vector3[2];
    public readonly float[] distance = new float[2];
    public bool active;
}

public class RaceReplay : MonoBehaviour {
    ReplayData replay;
    GameObject[] cars;
    Camera chaseCamera;
    Vector3 cameraVelocity;
    Vector3 smoothedForward;
    Vector3 previousEgoPosition;
    bool cameraInitialized;
    float smoothedFov = 55f;
    TireTrail[] tireTrails;
    RenderTexture captureTarget;
    Texture2D captureTexture;
    string outputDirectory;
    bool capture;

    [RuntimeInitializeOnLoadMethod]
    static void StartReplay() {
        new GameObject("RaceReplay").AddComponent<RaceReplay>();
    }

    void Start() {
        string replayPath = Argument("--replay");
        outputDirectory = Argument("--frames");
        capture = !String.IsNullOrEmpty(outputDirectory);
        if (String.IsNullOrEmpty(replayPath) || !File.Exists(replayPath)) {
            Debug.LogError("Missing --replay <json> argument");
            Application.Quit(2);
            return;
        }
        replay = JsonUtility.FromJson<ReplayData>(File.ReadAllText(replayPath));
        if (replay == null || replay.frames == null || replay.frames.Length == 0) {
            Debug.LogError("Replay contains no frames");
            Application.Quit(3);
            return;
        }
        if (capture) Directory.CreateDirectory(outputDirectory);
        BuildWorld();
        StartCoroutine(Play());
    }

    string Argument(string name) {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++) if (args[i] == name) return args[i + 1];
        return "";
    }

    Material Material(Color color) {
        Material template = Resources.Load<Material>("ReplayMaterial");
        if (template == null) throw new Exception("Bundled ReplayMaterial is missing");
        Material material = new Material(template);
        material.color = color;
        return material;
    }

    Material LitMaterial(Color color) {
        Material template = Resources.Load<Material>("ReplayLitTemplate");
        if (template == null) return Material(color);
        Material material = new Material(template);
        material.color = color;
        return material;
    }

    void BuildWorld() {
        QualitySettings.antiAliasing = 4;
        QualitySettings.shadowDistance = 90f;
        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.48f, 0.52f, 0.58f);
        RenderSettings.skybox = Resources.Load<Material>("ReplaySky");
        RenderSettings.defaultReflectionMode = DefaultReflectionMode.Skybox;
        RenderSettings.reflectionIntensity = 1.15f;
        RenderSettings.fog = true;
        RenderSettings.fogColor = new Color(0.68f, 0.76f, 0.83f);
        RenderSettings.fogMode = FogMode.Linear;
        RenderSettings.fogStartDistance = 45f;
        RenderSettings.fogEndDistance = 135f;
        GameObject lightObject = new GameObject("Sun");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.15f;
        light.shadows = LightShadows.Soft;
        lightObject.transform.rotation = Quaternion.Euler(42f, -35f, 0f);

        BuildTrack();
        Color[] colors = { new Color(0.28f, 0.78f, 1f), new Color(1f, 0.48f, 0.3f), new Color(1f, 0.9f, 0.28f) };
        string[] models = { "ReplayAssets/RaceCar", "ReplayAssets/FutureCar", "ReplayAssets/RaceCar" };
        cars = new GameObject[3];
        for (int i = 0; i < cars.Length; i++) cars[i] = BuildCar("Car " + (i + 1), colors[i], models[i]);
        BuildTireTrails();

        GameObject cameraObject = new GameObject("Third Person Chase Camera");
        chaseCamera = cameraObject.AddComponent<Camera>();
        chaseCamera.tag = "MainCamera";
        chaseCamera.clearFlags = CameraClearFlags.Skybox;
        chaseCamera.fieldOfView = 62f;
        chaseCamera.nearClipPlane = 0.05f;
        chaseCamera.farClipPlane = 250f;
        if (capture) {
            captureTarget = new RenderTexture(960, 540, 24, RenderTextureFormat.ARGB32);
            captureTarget.name = "Offline capture target";
            captureTexture = new Texture2D(960, 540, TextureFormat.RGB24, false);
        }
    }

    Bounds TrackBounds() {
        Bounds bounds = new Bounds(new Vector3(replay.trackX[0], 0, replay.trackY[0]), Vector3.zero);
        for (int i = 1; i < replay.trackX.Length; i++) bounds.Encapsulate(new Vector3(replay.trackX[i], 0, replay.trackY[i]));
        return bounds;
    }

    void BuildTrack() {
        GameObject grass = GameObject.CreatePrimitive(PrimitiveType.Plane);
        grass.name = "Ground";
        Bounds bounds = TrackBounds();
        grass.transform.position = new Vector3(bounds.center.x, -0.06f, bounds.center.y);
        grass.transform.localScale = new Vector3(Mathf.Max(5f, bounds.size.x / 10f + 1f), 1f, Mathf.Max(5f, bounds.size.y / 10f + 1f));
        Material groundMaterial = new Material(Resources.Load<Material>("ReplayGround"));
        groundMaterial.mainTextureScale = new Vector2(Mathf.Max(3f, bounds.size.x / 4f), Mathf.Max(3f, bounds.size.y / 4f));
        grass.GetComponent<Renderer>().material = groundMaterial;

        BuildRoadMesh();
        BuildRaceline();

    }

    void BuildRaceline() {
        if (replay.racelineX == null || replay.racelineY == null || replay.racelineX.Length < 2) return;
        GameObject lineObject = new GameObject("Recorded raceline");
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        line.loop = true;
        line.useWorldSpace = true;
        line.positionCount = replay.racelineX.Length;
        line.startWidth = 0.025f;
        line.endWidth = 0.025f;
        line.numCornerVertices = 3;
        line.numCapVertices = 2;
        line.material = Material(new Color(0.03f, 0.42f, 1f));
        for (int i = 0; i < replay.racelineX.Length; i++) {
            line.SetPosition(i, new Vector3(replay.racelineX[i], 0.032f, replay.racelineY[i]));
        }
    }

    void BuildRoadMesh() {
        int count = replay.trackX.Length;
        Vector3[] vertices = new Vector3[(count + 1) * 2];
        Vector2[] uv = new Vector2[(count + 1) * 2];
        int[] triangles = new int[count * 6];
        float halfWidth = replay.trackWidth * 0.5f;
        float distance = 0f;
        for (int ring = 0; ring <= count; ring++) {
            int i = ring % count;
            int prev = (i - 1 + count) % count;
            int next = (i + 1) % count;
            Vector2 tangent = new Vector2(replay.trackX[next] - replay.trackX[prev], replay.trackY[next] - replay.trackY[prev]).normalized;
            Vector2 normal = new Vector2(-tangent.y, tangent.x);
            vertices[2 * ring] = new Vector3(replay.trackX[i] + normal.x * halfWidth, 0f, replay.trackY[i] + normal.y * halfWidth);
            vertices[2 * ring + 1] = new Vector3(replay.trackX[i] - normal.x * halfWidth, 0f, replay.trackY[i] - normal.y * halfWidth);
            uv[2 * ring] = new Vector2(0f, distance / 1.8f);
            uv[2 * ring + 1] = new Vector2(1f, distance / 1.8f);
            if (ring < count) {
                int following = (i + 1) % count;
                distance += Vector2.Distance(new Vector2(replay.trackX[i], replay.trackY[i]), new Vector2(replay.trackX[following], replay.trackY[following]));
            }
        }
        for (int i = 0; i < count; i++) {
            int t = i * 6;
            triangles[t] = 2 * i;
            triangles[t + 1] = 2 * (i + 1);
            triangles[t + 2] = 2 * i + 1;
            triangles[t + 3] = 2 * i + 1;
            triangles[t + 4] = 2 * (i + 1);
            triangles[t + 5] = 2 * (i + 1) + 1;
        }
        Mesh mesh = new Mesh();
        mesh.name = "Procedural race track";
        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        GameObject road = new GameObject("Track surface");
        road.AddComponent<MeshFilter>().mesh = mesh;
        road.AddComponent<MeshRenderer>().material = Resources.Load<Material>("ReplayRoad");
    }

    GameObject BuildCar(string name, Color color, string modelResource) {
        GameObject root = new GameObject(name);
        GameObject model = Resources.Load<GameObject>(modelResource);
        if (model != null) {
            GameObject holder = new GameObject("Recorded vehicle dimensions");
            holder.transform.SetParent(root.transform, false);
            GameObject visual = Instantiate(model, holder.transform);
            visual.name = "Kenney vehicle";
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localRotation = Quaternion.Euler(0f, 90f, 0f);
            visual.transform.localScale = Vector3.one;
            Bounds modelBounds = RendererBounds(visual);
            if (modelBounds.size.x > 0.001f) {
                visual.transform.localScale = Vector3.one * (replay.vehicleLength / modelBounds.size.x);
            }
            modelBounds = RendererBounds(visual);
            Vector3 holderScale = Vector3.one;
            if (modelBounds.size.z > 0.001f) holderScale.z = replay.vehicleWidth / modelBounds.size.z;
            if (modelBounds.size.y > 0.001f) holderScale.y = Mathf.Min(1f, 0.16f / modelBounds.size.y);
            holder.transform.localScale = holderScale;
            modelBounds = RendererBounds(holder);
            holder.transform.position += Vector3.up * (0.015f - modelBounds.min.y);
            ApplyVehicleMaterials(visual, color);
            return root;
        }
        BuildFallbackCar(root, color);
        return root;
    }

    Bounds RendererBounds(GameObject target) {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return new Bounds(target.transform.position, Vector3.zero);
        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++) bounds.Encapsulate(renderers[i].bounds);
        return bounds;
    }

    void ApplyVehicleMaterials(GameObject vehicle, Color bodyColor) {
        foreach (Renderer renderer in vehicle.GetComponentsInChildren<Renderer>()) {
            string part = renderer.gameObject.name.ToLowerInvariant();
            Material material;
            if (part.Contains("wheel") || part.Contains("tire")) {
                material = LitMaterial(new Color(0.025f, 0.028f, 0.035f));
                material.SetFloat("_Metallic", 0f);
                material.SetFloat("_Glossiness", 0.18f);
            } else if (part.Contains("window") || part.Contains("glass")) {
                material = LitMaterial(new Color(0.06f, 0.16f, 0.24f));
                material.SetFloat("_Metallic", 0.15f);
                material.SetFloat("_Glossiness", 0.92f);
            } else if (part.Contains("light")) {
                material = LitMaterial(new Color(1f, 0.92f, 0.55f));
                material.SetFloat("_Glossiness", 0.8f);
            } else {
                material = LitMaterial(bodyColor);
                material.SetFloat("_Metallic", 0.62f);
                material.SetFloat("_Glossiness", 0.82f);
            }
            renderer.material = material;
        }
    }

    void BuildFallbackCar(GameObject root, Color color) {
        GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cube);
        body.name = "Body";
        body.transform.SetParent(root.transform, false);
        body.transform.localPosition = new Vector3(-0.015f, 0.12f, 0f);
        body.transform.localScale = new Vector3(0.44f, 0.11f, 0.20f);
        body.GetComponent<Renderer>().material = Material(color);
        GameObject nose = GameObject.CreatePrimitive(PrimitiveType.Cube);
        nose.name = "Nose";
        nose.transform.SetParent(root.transform, false);
        nose.transform.localPosition = new Vector3(0.255f, 0.085f, 0f);
        nose.transform.localScale = new Vector3(0.18f, 0.065f, 0.095f);
        nose.GetComponent<Renderer>().material = Material(color * 0.8f);
        GameObject cockpit = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        cockpit.name = "Cockpit";
        cockpit.transform.SetParent(root.transform, false);
        cockpit.transform.localPosition = new Vector3(-0.055f, 0.205f, 0f);
        cockpit.transform.localScale = new Vector3(0.18f, 0.12f, 0.13f);
        cockpit.GetComponent<Renderer>().material = Material(new Color(0.035f, 0.055f, 0.075f));
        AddWing(root.transform, "Front wing", new Vector3(0.34f, 0.065f, 0f), new Vector3(0.055f, 0.025f, 0.34f), color * 0.75f);
        AddWing(root.transform, "Rear wing", new Vector3(-0.27f, 0.19f, 0f), new Vector3(0.045f, 0.055f, 0.31f), color * 0.65f);
        float[] wheelX = { -0.16f, -0.16f, 0.17f, 0.17f };
        float[] wheelZ = { -0.145f, 0.145f, -0.145f, 0.145f };
        for (int i = 0; i < 4; i++) {
            GameObject wheel = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            wheel.name = "Wheel " + i;
            wheel.transform.SetParent(root.transform, false);
            wheel.transform.localPosition = new Vector3(wheelX[i], 0.075f, wheelZ[i]);
            wheel.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            wheel.transform.localScale = new Vector3(0.075f, 0.035f, 0.075f);
            wheel.GetComponent<Renderer>().material = Material(new Color(0.025f, 0.025f, 0.03f));
        }
    }

    void AddWing(Transform parent, string name, Vector3 position, Vector3 scale, Color color) {
        GameObject wing = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wing.name = name;
        wing.transform.SetParent(parent, false);
        wing.transform.localPosition = position;
        wing.transform.localScale = scale;
        wing.GetComponent<Renderer>().material = Material(color);
    }

    void BuildTireTrails() {
        tireTrails = new TireTrail[cars.Length];
        for (int car = 0; car < cars.Length; car++) {
            TireTrail trail = new TireTrail();
            trail.mesh = new Mesh();
            trail.mesh.name = "Throttle tire marks " + (car + 1);
            GameObject marks = new GameObject(trail.mesh.name);
            marks.AddComponent<MeshFilter>().mesh = trail.mesh;
            marks.AddComponent<MeshRenderer>().material = Resources.Load<Material>("ReplayTireMark");
            tireTrails[car] = trail;
        }
    }

    void UpdateTireTrail(int carIndex, float throttle) {
        TireTrail trail = tireTrails[carIndex];
        float rearX = -replay.vehicleLength * 0.42f;
        float wheelZ = replay.vehicleWidth * 0.375f;
        Vector3 left = cars[carIndex].transform.TransformPoint(new Vector3(rearX, 0.022f, -wheelZ));
        Vector3 right = cars[carIndex].transform.TransformPoint(new Vector3(rearX, 0.022f, wheelZ));
        bool marking = Mathf.Abs(throttle) >= 0.58f;
        if (marking && trail.active) {
            AddTireSegment(trail, 0, trail.previous[0], left);
            AddTireSegment(trail, 1, trail.previous[1], right);
            trail.mesh.SetVertices(trail.vertices);
            trail.mesh.SetUVs(0, trail.uv);
            trail.mesh.SetTriangles(trail.triangles, 0);
            trail.mesh.RecalculateBounds();
        }
        trail.previous[0] = left;
        trail.previous[1] = right;
        trail.active = marking;
    }

    void AddTireSegment(TireTrail trail, int wheel, Vector3 start, Vector3 end) {
        Vector3 delta = end - start;
        delta.y = 0f;
        if (delta.sqrMagnitude < 0.000025f) return;
        Vector3 side = Vector3.Cross(Vector3.up, delta.normalized) * 0.018f;
        int v = trail.vertices.Count;
        float startV = trail.distance[wheel] / 0.065f;
        trail.distance[wheel] += delta.magnitude;
        float endV = trail.distance[wheel] / 0.065f;
        trail.vertices.Add(start - side);
        trail.vertices.Add(start + side);
        trail.vertices.Add(end - side);
        trail.vertices.Add(end + side);
        trail.uv.Add(new Vector2(0f, startV));
        trail.uv.Add(new Vector2(1f, startV));
        trail.uv.Add(new Vector2(0f, endV));
        trail.uv.Add(new Vector2(1f, endV));
        trail.triangles.Add(v); trail.triangles.Add(v + 2); trail.triangles.Add(v + 1);
        trail.triangles.Add(v + 1); trail.triangles.Add(v + 2); trail.triangles.Add(v + 3);
    }

    void UpdateChaseCamera(float yaw) {
        float dt = 1f / Mathf.Max(1f, replay.fps);
        Vector3 target = cars[0].transform.position;
        Vector3 rawForward = new Vector3(Mathf.Cos(yaw), 0f, Mathf.Sin(yaw));
        if (!cameraInitialized) {
            smoothedForward = rawForward;
            previousEgoPosition = target;
        } else {
            float directionBlend = 1f - Mathf.Exp(-6f * dt);
            smoothedForward = Vector3.Slerp(smoothedForward, rawForward, directionBlend).normalized;
        }
        float speed = cameraInitialized ? Vector3.Distance(target, previousEgoPosition) / dt : 0f;
        previousEgoPosition = target;
        float chaseDistance = 1.55f + Mathf.Min(0.55f, speed * 0.04f);
        float chaseHeight = 0.78f + Mathf.Min(0.2f, speed * 0.01f);
        Vector3 desiredPosition = target - smoothedForward * chaseDistance + Vector3.up * chaseHeight;
        Vector3 lookPoint = target + smoothedForward * (0.9f + Mathf.Min(0.7f, speed * 0.04f)) + Vector3.up * 0.16f;
        Quaternion desiredRotation = Quaternion.LookRotation(lookPoint - desiredPosition, Vector3.up);
        if (!cameraInitialized) {
            chaseCamera.transform.position = desiredPosition;
            chaseCamera.transform.rotation = desiredRotation;
            smoothedFov = 55f;
            cameraInitialized = true;
        } else {
            chaseCamera.transform.position = Vector3.SmoothDamp(
                chaseCamera.transform.position, desiredPosition, ref cameraVelocity, 0.2f, 20f, dt
            );
            float rotationBlend = 1f - Mathf.Exp(-8f * dt);
            chaseCamera.transform.rotation = Quaternion.Slerp(
                chaseCamera.transform.rotation, desiredRotation, rotationBlend
            );
            float targetFov = 55f + Mathf.Min(9f, speed * 0.65f);
            smoothedFov = Mathf.Lerp(smoothedFov, targetFov, 1f - Mathf.Exp(-3.5f * dt));
        }
        chaseCamera.fieldOfView = smoothedFov;
    }

    IEnumerator Play() {
        Application.targetFrameRate = Mathf.RoundToInt(replay.fps);
        // Establish the first recorded pose before any splash-screen or capture frames.
        ApplyFrame(0);
        for (int warmup = 0; warmup < 3; warmup++) yield return new WaitForEndOfFrame();
        for (int frame = 0; frame < replay.frames.Length; frame++) {
            ApplyFrame(frame);
            if (capture) {
                CaptureOffscreen(Path.Combine(outputDirectory, String.Format("frame_{0:D6}.png", frame)));
                yield return null;
            } else {
                yield return new WaitForEndOfFrame();
                yield return new WaitForSecondsRealtime(1f / replay.fps);
            }
        }
        if (capture) {
            yield return new WaitForSecondsRealtime(0.5f);
            File.WriteAllText(Path.Combine(outputDirectory, "COMPLETE"), replay.frames.Length.ToString(CultureInfo.InvariantCulture));
            Application.Quit(0);
        }
    }

    void CaptureOffscreen(string path) {
        RenderTexture previous = RenderTexture.active;
        RenderTexture previousTarget = chaseCamera.targetTexture;
        chaseCamera.targetTexture = captureTarget;
        chaseCamera.Render();
        RenderTexture.active = captureTarget;
        captureTexture.ReadPixels(new Rect(0, 0, captureTarget.width, captureTarget.height), 0, 0, false);
        captureTexture.Apply(false);
        File.WriteAllBytes(path, captureTexture.EncodeToPNG());
        chaseCamera.targetTexture = previousTarget;
        RenderTexture.active = previous;
    }

    void OnDestroy() {
        if (captureTarget != null) captureTarget.Release();
        if (captureTexture != null) Destroy(captureTexture);
    }

    void ApplyFrame(int frame) {
        float[] pose = replay.frames[frame].poses;
        for (int car = 0; car < cars.Length; car++) {
            float x = pose[3 * car];
            float y = pose[3 * car + 1];
            float yaw = pose[3 * car + 2] * Mathf.Rad2Deg;
            cars[car].transform.position = new Vector3(x, 0f, y);
            cars[car].transform.rotation = Quaternion.Euler(0f, -yaw, 0f);
        }
        float[] throttle = replay.frames[frame].throttle;
        for (int car = 0; car < cars.Length; car++) {
            UpdateTireTrail(car, throttle != null && car < throttle.Length ? throttle[car] : 0f);
        }
        UpdateChaseCamera(pose[2]);
    }
}
