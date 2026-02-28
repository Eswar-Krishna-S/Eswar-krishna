# NumPy Universal Functions - Mechanical Engineering

## ?? Assignment Overview
- **Points:** 100 total
- **Topics:** NumPy universal functions, mechanical engineering applications
- **Submission:** Push your code to GitHub

---

## ?? Getting Started

### Step 1: Create Your Own Repository

1. Click the green **"Use this template"** button at the top of this page
2. Name your repository: `numpy-assignment-YOUR_NAME`
3. Make it **Private** if you don't want others to see your work
4. Click **"Create repository"**

### Step 2: Clone to Your Computer
```bash
git clone https://github.com/YOUR_USERNAME/numpy-assignment-YOUR_NAME.git
cd numpy-assignment-YOUR_NAME
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ?? Complete the Assignment

1. Open `mechanical_problems.py` in any text editor
2. Complete all 10 functions using **NumPy ufuncs only**
3. **NO Python loops allowed!** (no `for` or `while`)

### Rules:
- ? Use NumPy functions: `np.sqrt()`, `np.sin()`, `np.cos()`, etc.
- ? Work with arrays
- ? Use broadcasting
- ? NO loops (`for`, `while`)
- ? NO list comprehensions for computation

---

## ?? Submit Your Work
```bash
git add mechanical_problems.py
git commit -m "Complete assignment"
git push
```

After pushing, check the **Actions** tab in your repository to see your grade!

---

## ? Check Your Grade

1. Go to your repository on GitHub
2. Click **"Actions"** tab (near the top)
3. Click on the latest workflow run
4. See your results:
   - ? Green checkmark = PASSED = Got points
   - ? Red X = FAILED = 0 points for that section

### Point Distribution:
- Q1-Q2: Von Mises & Projectile ? 20 points
- Q3-Q4: Forces & Thermal ? 20 points
- Q5-Q6: Angular & Beam ? 20 points
- Q7-Q8: Velocity & Power ? 20 points
- Q9-Q10: Spring & Damping ? 20 points
- **Total: 100 points**

---

## ?? Questions

### Q1: Von Mises Stress (10 points)
Calculate von Mises stress from principal stresses.
- Formula: s_vm = v(s1² - s1s2 + s2²)

### Q2: Projectile Trajectory (10 points)
Calculate x, y coordinates for projectile motion.
- Equations: x = v0cos(?)t, y = v0sin(?)t - ½gt²

### Q3: Force Resultant (10 points)
Calculate magnitude and direction of resultant forces.

### Q4: Thermal Expansion (10 points)
Calculate length change due to thermal expansion.
- Formula: ?L = a × L0 × ?T

### Q5: Angular Velocity Conversion (10 points)
Convert RPM to rad/s and calculate angular displacement.

### Q6: Beam Deflection (10 points)
Calculate deflection of simply supported beam.

### Q7: Velocity Components (10 points)
Resolve velocities into x and y components.

### Q8: Power to Torque (10 points)
Calculate torque from power and angular velocity.
- Formula: t = P / ?

### Q9: Spring System (10 points)
Calculate spring force and potential energy.
- Formulas: F = -kx, U = ½kx²

### Q10: Damped Oscillation (10 points)
Calculate displacement for damped harmonic motion.
- Formula: x(t) = A × e^(-bt) × cos(?t)

---

## ?? Troubleshooting

**Tests fail:**
- Make sure all `pass` statements are replaced with code
- Check you're using NumPy functions, not loops
- Verify angles are converted to radians where needed

**Can't push to GitHub:**
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
```

**Import errors:**
```bash
pip install numpy pytest
```

---

## ?? Resources

- NumPy Documentation: https://numpy.org/doc/stable/
- NumPy Ufuncs: https://numpy.org/doc/stable/reference/ufuncs.html

---

Good luck! ??

--