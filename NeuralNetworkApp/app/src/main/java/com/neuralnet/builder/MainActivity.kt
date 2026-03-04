package com.neuralnet.builder

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.neuralnet.builder.builder.NetworkBuilderActivity
import com.neuralnet.builder.databinding.ActivityMainBinding
import com.neuralnet.builder.tutorial.TutorialActivity

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupClickListeners()
    }

    private fun setupClickListeners() {
        binding.btnLearn.setOnClickListener {
            startActivity(Intent(this, TutorialActivity::class.java))
        }

        binding.btnBuild.setOnClickListener {
            startActivity(Intent(this, NetworkBuilderActivity::class.java))
        }

        binding.btnQuickStart.setOnClickListener {
            // Launch builder with a preset beginner config
            val intent = Intent(this, NetworkBuilderActivity::class.java)
            intent.putExtra(NetworkBuilderActivity.EXTRA_QUICK_START, true)
            startActivity(intent)
        }
    }
}
