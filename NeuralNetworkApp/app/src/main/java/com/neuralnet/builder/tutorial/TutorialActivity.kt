package com.neuralnet.builder.tutorial

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.neuralnet.builder.R
import com.neuralnet.builder.databinding.ActivityTutorialBinding
import com.neuralnet.builder.databinding.ItemTutorialPageBinding
import com.neuralnet.builder.databinding.ItemChapterBinding

class TutorialActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTutorialBinding
    private var currentChapterIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTutorialBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupChapterList()
        loadChapter(0)
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Tutorials"
        binding.toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }
    }

    private fun setupChapterList() {
        val chapters = TutorialData.chapters
        binding.chapterRecycler.layoutManager =
            LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        binding.chapterRecycler.adapter = ChapterTabAdapter(chapters) { index ->
            currentChapterIndex = index
            loadChapter(index)
        }
    }

    private fun loadChapter(chapterIndex: Int) {
        val chapter = TutorialData.chapters.getOrNull(chapterIndex) ?: return

        binding.chapterTitle.text = "${chapter.icon}  ${chapter.title}"
        binding.pageIndicator.text = "1 / ${chapter.pages.size}"

        val adapter = TutorialPageAdapter(chapter.pages)
        binding.viewPager.adapter = adapter
        binding.viewPager.setCurrentItem(0, false)

        binding.viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                binding.pageIndicator.text = "${position + 1} / ${chapter.pages.size}"
                binding.prevBtn.isEnabled = position > 0
                binding.nextBtn.isEnabled = position < chapter.pages.size - 1
            }
        })

        binding.prevBtn.isEnabled = false
        binding.nextBtn.isEnabled = chapter.pages.size > 1

        binding.prevBtn.setOnClickListener {
            val cur = binding.viewPager.currentItem
            if (cur > 0) binding.viewPager.setCurrentItem(cur - 1, true)
        }
        binding.nextBtn.setOnClickListener {
            val cur = binding.viewPager.currentItem
            if (cur < chapter.pages.size - 1) {
                binding.viewPager.setCurrentItem(cur + 1, true)
            } else {
                // Move to next chapter
                val next = currentChapterIndex + 1
                if (next < TutorialData.chapters.size) {
                    currentChapterIndex = next
                    loadChapter(next)
                    binding.chapterRecycler.scrollToPosition(next)
                }
            }
        }
    }
}

// ─── Chapter Tab Adapter ──────────────────────────────────────────────────────

class ChapterTabAdapter(
    private val chapters: List<TutorialChapter>,
    private val onSelect: (Int) -> Unit
) : RecyclerView.Adapter<ChapterTabAdapter.VH>() {

    private var selected = 0

    inner class VH(val binding: ItemChapterBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        VH(ItemChapterBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun getItemCount() = chapters.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val ch = chapters[position]
        holder.binding.chapterIcon.text  = ch.icon
        holder.binding.chapterLabel.text = ch.title
        holder.binding.root.isSelected   = position == selected
        holder.binding.root.setOnClickListener {
            val prev = selected
            selected = position
            notifyItemChanged(prev)
            notifyItemChanged(selected)
            onSelect(position)
        }
    }
}

// ─── Tutorial Page Adapter ────────────────────────────────────────────────────

class TutorialPageAdapter(
    private val pages: List<TutorialPage>
) : RecyclerView.Adapter<TutorialPageAdapter.VH>() {

    inner class VH(val binding: ItemTutorialPageBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        VH(ItemTutorialPageBinding.inflate(LayoutInflater.from(parent.context), parent, false))

    override fun getItemCount() = pages.size

    override fun onBindViewHolder(holder: VH, position: Int) {
        val page = pages[position]
        holder.binding.pageEmoji.text = page.emoji
        holder.binding.pageTitle.text = page.title
        holder.binding.pageBody.text  = page.body

        if (page.tip.isNotBlank()) {
            holder.binding.tipCard.visibility = View.VISIBLE
            holder.binding.tipText.text = "💡 ${page.tip}"
        } else {
            holder.binding.tipCard.visibility = View.GONE
        }

        // Show visual placeholder based on visual type
        holder.binding.visualContainer.visibility =
            if (page.visual != TutorialVisual.NONE) View.VISIBLE else View.GONE
        if (page.visual != TutorialVisual.NONE) {
            holder.binding.visualLabel.text = visualDescription(page.visual)
        }
    }

    private fun visualDescription(v: TutorialVisual) = when (v) {
        TutorialVisual.NEURON_DIAGRAM  -> "[ Neuron Diagram: inputs → weighted sum → activation → output ]"
        TutorialVisual.LAYER_DIAGRAM   -> "[ Layer Diagram: Input Layer → Hidden Layer → Output Layer ]"
        TutorialVisual.NETWORK_DIAGRAM -> "[ Network: 4 input nodes connected through 2 hidden layers to 1 output ]"
        TutorialVisual.TRAINING_LOOP   -> "[ Training Loop: Forward Pass → Loss → Backprop → Update → repeat ]"
        TutorialVisual.ACTIVATION_GRAPH-> "[ Activation Graphs: ReLU / Sigmoid / Tanh curves plotted ]"
        TutorialVisual.LOSS_CURVE      -> "[ Loss Curve: Training loss decreasing over epochs ]"
        TutorialVisual.NONE            -> ""
    }
}
