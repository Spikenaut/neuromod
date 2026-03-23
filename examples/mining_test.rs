//! Example: Mining Reward Integration Test
//! 
//! Demonstrates the new mining_dopamine field and MiningReward functionality

use neuromod::{NeuroModulators, MiningReward};

fn main() {
    println!("🦁 Testing neuromod v0.2.1 Mining Integration");
    
    // Test 1: Create default neuromodulators with mining_dopamine
    let mut modulators = NeuroModulators::default();
    println!("✅ Default NeuroModulators:");
    println!("   dopamine: {:.3}", modulators.dopamine);
    println!("   cortisol: {:.3}", modulators.cortisol);
    println!("   acetylcholine: {:.3}", modulators.acetylcholine);
    println!("   tempo: {:.3}", modulators.tempo);
    println!("   mining_dopamine: {:.3} ← NEW!", modulators.mining_dopamine);
    
    // Test 2: Create mining reward calculator
    let mut mining_reward = MiningReward::new();
    
    // Simulate mining telemetry (good conditions)
    let reward = mining_reward.compute(1.2, 350.0, 72.0); // hashrate, power, temp
    println!("✅ Mining reward (good conditions): {:.3}", reward);
    
    // Add mining reward to neuromodulators
    modulators.mining_dopamine = reward;
    println!("✅ Updated mining_dopamine: {:.3}", modulators.mining_dopamine);
    
    // Test 3: Apply decay (homeostasis)
    modulators.decay();
    println!("✅ After decay - mining_dopamine: {:.3}", modulators.mining_dopamine);
    
    // Test 4: Check mining reward status
    if modulators.is_mining_rewarded() {
        println!("✅ Mining is rewarding!");
    } else {
        println!("⚠️  Mining needs improvement");
    }
    
    println!("🎯 neuromod v0.2.1 mining integration test complete!");
}
