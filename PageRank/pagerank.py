import os
import random
import re
import sys
import numpy as np
DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    next_node_probability = {}
    connected_links = corpus.get(page, set())
    all_links = list(corpus.keys())

    if connected_links:
        for link in all_links:
            next_node_probability[link] = (1 - damping_factor) / len(all_links)
        for link in connected_links:
            next_node_probability[link] += damping_factor / len(connected_links)

    else:
        for link in all_links:
            next_node_probability[link] = 1 / len(all_links)

    return next_node_probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    all_pages = list(corpus.keys())
    page = random.choice(all_pages)
    pagerank[page] = 1/n
    for pg in all_pages:
        pagerank[pg] = 0
    for _ in range(n):
        transition = transition_model(corpus, page, damping_factor)
        page = random.choices(population=list(transition.keys()),
                              weights=list(transition.values()), k=1)[0]
        pagerank[page] += 1/n
    total = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    highest_diff = np.inf
    all_pages = list(corpus.keys())
    num_pages = len(all_pages)
    for page in all_pages:
        pagerank[page] = 1 / num_pages
    while highest_diff >= 0.001:
        highest_diff = 0
        for page in all_pages:
            new_pagerank = page_rank_formula(damping_factor, num_pages, page, corpus, pagerank)
            difference = abs(pagerank[page] - new_pagerank)
            if difference > highest_diff:
                highest_diff = difference
            pagerank[page] = new_pagerank
    total = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total
    return pagerank


def page_rank_formula(damping_factor, number_of_pages, page, corpus, pagerank):
    """
    Return PageRank value for a page using the PageRank formula.
    """
    link_sum = 0
    for link in corpus:
        if page in corpus[link]:
            link_sum += pagerank[link] / len(corpus[link])
    new_pagerank = ((1 - damping_factor) / number_of_pages) + (damping_factor * link_sum)
    return new_pagerank


if __name__ == "__main__":
    main()
