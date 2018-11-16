class SkipList:
	def __init__(self, postings_file, offset):
		self.postings_file = postings_file
		self.offset = offset
		self.current_ptr = 0 # stores current pointer position
		self.eol = False # check if the end of list is reached
		self.count = 0 # used as a utility counter variable

	def peek(self):
		# Peeks at next value in the list
		if self.eol:
			return "EOL"

		current_offset = self.current_ptr
		value = self.get_value(current_offset, False)
		current_offset += self.count
		if value[0]:
			skip_distance = int(value[1])
			return (True, self.get_value(current_offset + skip_distance, False)[1], current_offset + skip_distance)

		return (False, value[1])

	def next(self):
		# Shifts current pointer and retrieve next value in list
		if self.eol:
			return "EOL"

		current_offset = self.current_ptr
		value = self.get_value(current_offset, True)
		current_offset += self.count
		self.current_ptr = current_offset
		if value[0]:
			skip_distance = int(value[1])
			return (True, self.get_value(current_offset + skip_distance, False)[1], current_offset + skip_distance)

		return (False, value[1])

	def get_value(self, current_offset, is_next):
		# Retrieves one posting entry from current_offset
		# Returns true if it is a skip pointer (skip value is returned)

		self.count = 0
		value = ""
		while 1:
			self.postings_file.seek(self.offset + current_offset)
			char = self.postings_file.read(1)
			if char == " ":
				self.count += 1
				break
			if char == "\n":
				if is_next:
					self.eol = True
				break

			value += char
			current_offset += 1
			self.count += 1

		if "+" in value:
			# strip prefix and return skip value
			return (True, value[1:])

		value = value.split(":")
		return (False, (int(value[0]), value[1], self.construct_position_list(value[2].split(","))))

	def skip(self, new_offset):
		self.current_ptr = new_offset

	#",".join([str(p - post[2][i-1]) if i > 0 else str(p) for i, p in enumerate(post[2])])
	def construct_position_list(self, list):
	    curr_sum = int(list[0])
	    construct = []
	    for i, p in enumerate(list):
	        if i == 0:
	            construct.append(int(p))
	        else:
	            curr_sum += int(p)
	            construct.append(curr_sum)

	    return construct

	def to_list(self):
		# Converts the postings list to list form

		saved_current = self.current_ptr
		saved_eol = self.eol

		postings_list = []
		self.current_ptr = 0
		self.eol = False

		while self.peek() != "EOL":
			value = self.peek()
			if value[0] == False:
				postings_list.append(value[1])
			self.next()

		self.current_ptr = saved_current
		self.eol = saved_eol
		return postings_list

	'''Assumption: only queried when peek() does not return EOL'''
	def skippable(self):
		return self.peek()[0]

	def index(self):
	    return self.peek()[1][0]

	def tf(self):
	    return self.peek()[1][1]

	def position(self):
	    return self.peek()[1][2]

	def value(self):
		return self.peek()[1]

	def skip_offset(self):
	    return self.peek()[2]
