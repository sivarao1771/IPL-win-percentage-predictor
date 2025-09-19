/**
 * 3D Background Animation for IPL Win Predictor
 * Creates an animated cricket stadium-like environment with floating elements
 */

class Background3D {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.cricketBalls = [];
        this.stumps = [];
        this.particles = [];
        this.animationId = null;
        
        this.init();
        this.createElements();
        this.setupLighting();
        this.animate();
        this.handleResize();
    }

    /**
     * Initialize the Three.js scene, camera, and renderer
     */
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(0x0a0a0a, 50, 200);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        this.camera.position.set(0, 5, 15);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: document.getElementById('bg-canvas'), 
            alpha: true,
            antialias: true 
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    /**
     * Create 3D elements for the background
     */
    createElements() {
        this.createCricketBalls();
        this.createStumps();
        this.createParticleField();
        this.createFieldBoundary();
    }

    /**
     * Create floating cricket balls with different colors representing different teams
     */
    createCricketBalls() {
        const ballColors = [
            0xff6b6b, // Red
            0x4ecdc4, // Teal  
            0x45b7d1, // Blue
            0x96ceb4, // Green
            0xfeca57, // Yellow
            0xff9ff3  // Pink
        ];

        for (let i = 0; i < 8; i++) {
            // Create ball geometry and material
            const geometry = new THREE.SphereGeometry(0.8, 16, 16);
            const material = new THREE.MeshPhysicalMaterial({
                color: ballColors[i % ballColors.length],
                roughness: 0.3,
                metalness: 0.1,
                clearcoat: 0.5,
                clearcoatRoughness: 0.1
            });
            
            const ball = new THREE.Mesh(geometry, material);
            
            // Position balls randomly in 3D space
            ball.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 40
            );
            
            // Add random rotation
            ball.rotation.set(
                Math.random() * Math.PI * 2,
                Math.random() * Math.PI * 2,
                Math.random() * Math.PI * 2
            );
            
            // Store initial position and add movement properties
            ball.userData = {
                initialPosition: ball.position.clone(),
                rotationSpeed: {
                    x: (Math.random() - 0.5) * 0.02,
                    y: (Math.random() - 0.5) * 0.02,
                    z: (Math.random() - 0.5) * 0.02
                },
                floatSpeed: Math.random() * 0.01 + 0.005,
                floatRange: Math.random() * 3 + 2
            };
            
            ball.castShadow = true;
            ball.receiveShadow = true;
            
            this.cricketBalls.push(ball);
            this.scene.add(ball);
        }
    }

    /**
     * Create cricket stumps scattered around the scene
     */
    createStumps() {
        for (let i = 0; i < 6; i++) {
            const stumpGroup = new THREE.Group();
            
            // Create three stumps
            for (let j = 0; j < 3; j++) {
                const stumpGeometry = new THREE.CylinderGeometry(0.05, 0.05, 2, 8);
                const stumpMaterial = new THREE.MeshLambertMaterial({ color: 0xdeb887 });
                const stump = new THREE.Mesh(stumpGeometry, stumpMaterial);
                
                stump.position.x = (j - 1) * 0.15;
                stump.position.y = 1;
                stump.castShadow = true;
                stumpGroup.add(stump);
            }
            
            // Add bails on top
            for (let j = 0; j < 2; j++) {
                const bailGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.15, 6);
                const bailMaterial = new THREE.MeshLambertMaterial({ color: 0xdeb887 });
                const bail = new THREE.Mesh(bailGeometry, bailMaterial);
                
                bail.position.x = (j - 0.5) * 0.15;
                bail.position.y = 2.05;
                bail.rotation.z = Math.PI / 2;
                bail.castShadow = true;
                stumpGroup.add(bail);
            }
            
            // Position stump groups randomly
            stumpGroup.position.set(
                (Math.random() - 0.5) * 60,
                -2,
                (Math.random() - 0.5) * 60
            );
            stumpGroup.rotation.y = Math.random() * Math.PI * 2;
            
            // Add floating animation properties
            stumpGroup.userData = {
                initialY: stumpGroup.position.y,
                floatSpeed: Math.random() * 0.005 + 0.002,
                floatRange: 0.5
            };
            
            this.stumps.push(stumpGroup);
            this.scene.add(stumpGroup);
        }
    }

    /**
     * Create a field of floating particles for atmosphere
     */
    createParticleField() {
        const particleCount = 1000;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const sizes = new Float32Array(particleCount);
        
        // Define particle colors (cricket team colors)
        const colorPalette = [
            new THREE.Color(0x667eea), // Blue
            new THREE.Color(0x764ba2), // Purple
            new THREE.Color(0xf093fb), // Pink
            new THREE.Color(0xf5576c), // Red
            new THREE.Color(0x4facfe), // Light Blue
            new THREE.Color(0x00f2fe)  // Cyan
        ];
        
        for (let i = 0; i < particleCount; i++) {
            // Position particles randomly in a large sphere
            const radius = Math.random() * 100 + 50;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            
            positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = radius * Math.cos(phi);
            positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
            
            // Assign random colors from palette
            const color = colorPalette[Math.floor(Math.random() * colorPalette.length)];
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
            
            // Random particle sizes
            sizes[i] = Math.random() * 2 + 0.5;
        }
        
        const particleGeometry = new THREE.BufferGeometry();
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        const particleMaterial = new THREE.PointsMaterial({
            size: 1,
            sizeAttenuation: true,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        this.particles = new THREE.Points(particleGeometry, particleMaterial);
        this.scene.add(this.particles);
    }

    /**
     * Create a visual field boundary with glowing lines
     */
    createFieldBoundary() {
        const points = [];
        const radius = 25;
        const segments = 64;
        
        for (let i = 0; i <= segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            points.push(new THREE.Vector3(
                Math.cos(angle) * radius,
                -1,
                Math.sin(angle) * radius
            ));
        }
        
        const boundaryGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const boundaryMaterial = new THREE.LineBasicMaterial({
            color: 0x4facfe,
            transparent: true,
            opacity: 0.3
        });
        
        const boundary = new THREE.Line(boundaryGeometry, boundaryMaterial);
        this.scene.add(boundary);
    }

    /**
     * Setup lighting for the scene
     */
    setupLighting() {
        // Ambient light for overall scene illumination
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Main directional light (like stadium lights)
        const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        mainLight.position.set(10, 20, 10);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        mainLight.shadow.camera.near = 0.1;
        mainLight.shadow.camera.far = 100;
        mainLight.shadow.camera.left = -50;
        mainLight.shadow.camera.right = 50;
        mainLight.shadow.camera.top = 50;
        mainLight.shadow.camera.bottom = -50;
        this.scene.add(mainLight);
        
        // Colored accent lights (team colors)
        const accentLight1 = new THREE.PointLight(0x667eea, 0.5, 30);
        accentLight1.position.set(-15, 10, -15);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xf093fb, 0.5, 30);
        accentLight2.position.set(15, 10, 15);
        this.scene.add(accentLight2);
        
        // Ground plane for shadows
        const groundGeometry = new THREE.PlaneGeometry(200, 200);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x0a0a0a, 
            transparent: true, 
            opacity: 0.1 
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -3;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    /**
     * Animation loop for all 3D elements
     */
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const time = Date.now() * 0.001;
        
        // Animate cricket balls
        this.cricketBalls.forEach((ball) => {
            const userData = ball.userData;
            
            // Floating motion
            ball.position.y = userData.initialPosition.y + 
                Math.sin(time * userData.floatSpeed) * userData.floatRange;
            
            // Continuous rotation
            ball.rotation.x += userData.rotationSpeed.x;
            ball.rotation.y += userData.rotationSpeed.y;
            ball.rotation.z += userData.rotationSpeed.z;
            
            // Subtle orbital motion
            ball.position.x = userData.initialPosition.x + 
                Math.cos(time * 0.1) * 2;
            ball.position.z = userData.initialPosition.z + 
                Math.sin(time * 0.1) * 2;
        });
        
        // Animate stumps
        this.stumps.forEach((stumpGroup) => {
            const userData = stumpGroup.userData;
            stumpGroup.position.y = userData.initialY + 
                Math.sin(time * userData.floatSpeed) * userData.floatRange;
        });
        
        // Animate particles
        if (this.particles) {
            this.particles.rotation.y += 0.0005;
            this.particles.rotation.x += 0.0002;
        }
        
        // Camera subtle movement for dynamic feel
        this.camera.position.x = Math.cos(time * 0.1) * 2;
        this.camera.position.y = 5 + Math.sin(time * 0.15) * 1;
        this.camera.lookAt(0, 0, 0);
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Handle window resize events
     */
    handleResize() {
        window.addEventListener('resize', () => {
            // Update camera aspect ratio
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            
            // Update renderer size
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        });
    }

    /**
     * Clean up resources when needed
     */
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Dispose of geometries and materials
        this.scene.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
        
        this.renderer.dispose();
    }
}

// Initialize the 3D background when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Background3D();
});